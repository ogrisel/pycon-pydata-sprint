import os
import sys
import socket
from collections import defaultdict
from hashlib import md5

from IPython import parallel

@parallel.util.interactive
def load_memmap(name, fname, dtype, shape):
    import numpy
    X = numpy.memmap(fname, shape=shape, dtype=dtype, mode='copyonwrite')
    globals().update({name:X})

# this fix is pending in master (0.13)
@parallel.util.interactive
def _push(**ns):
    globals().update(ns)

def default_identify():
    import socket
    return socket.gethostname(), "/tmp"


def save_for_memmap(A, fname):
    """save array A as memmapped array"""
    import os.path
    import numpy
    
    if os.path.exists(fname):
        # file already exists, nothing to do
        return
    
    mmA = numpy.memmap(fname, mode='w+', dtype=A.dtype, shape=A.shape)
    mmA[:] = A
    # maybe need to flush?
    # mmA.flush()

def datastore_mapping(view, identify_f):
    """generate various mappings of datastores and paths"""
    mapping = dv.apply_async(identify_f).get_dict()
    here, _ = identify_f()
    
    # reverse mapping, so we have a list of engine IDs per datastore
    revmap = defaultdict(list)
    paths = {}
    for eid, (datastore_id, path) in mapping.iteritems():
        revmap[datastore_id].append(eid)
        paths[datastore_id] = path
    
    return here, revmap, paths

def bcast_memmap(view, name, X, identify_f=default_identify):
    """
    
    identify_f: 
        callable that returns string identifier of engine
        local datastore.  An explicit push will be performed
        on exactly one engine per unique datastore id.
    """
    client = view.client
    
    here, revmap, paths = datastore_mapping(view, identify_f)

    ars = []
    mm_ars = []
    
    # checksum array for filename
    checksum = md5(X).hexdigest()
    
    # perform push to first engine of each non-local datastore:
    for datastore_id, targets in revmap.iteritems():
        if datastore_id != here:
            # push to first target at datastore
            e0 = client[targets[0]]
            ar = e0.apply(_push, **{name : X})
            # DEBUG:
            # ar.get()
            ars.append(ar)
            
            targets = targets[1:]
            # Nothing left to do if only one engine on this machine
            if not targets:
                continue
            
            fname = os.path.join(paths[datastore_id], checksum)
            # save to file for memmapping on other engines
            ar = e0.apply_async(save_for_memmap, parallel.Reference(name), fname)
            ars.append(ar)
            mm_ars.append(ar)
    
    # wait on pushes, to ensure files are ready for memmap in next step
    # this could be done without blocking the client with a barrier
    # (MPI or ZMQ or file lock, etc.)
    [ ar.get() for ar in mm_ars ]
    
    # loop through datastores
    for datastore_id, targets in revmap.iteritems():
        if datastore_id != here:
            targets = targets[1:]
            # Nothing left to do if only one engine on this machine
            if not targets:
                continue
            
            fname = os.path.join(paths[datastore_id], checksum)
            # load from memmapped file on engines after the first for this datastore
            other = client[targets]
            ar = other.apply_async(load_memmap, name, fname, X.dtype, X.shape)
            # DEBUG:
            # ar.get()
            ars.append(ar)
            
        else:
            if not isinstance(X, numpy.memmap):
                fname = os.path.join(paths[datastore_id], checksum)
                save_for_memmap(X, fname)
            else:
                fname = X.filename
            # local engines, load from original memmapped file
            ar = client[targets].apply_async(load_memmap, name, fname, X.dtype, X.shape)
            # DEBUG:
            # ar.get()
            ars.append(ar)
            
    return ars, revmap

if __name__ == '__main__':
    import socket
    
    rc = parallel.Client(profile='vm')
    dv = rc[:]
    with dv.sync_imports():
        import numpy

    A = numpy.memmap("/tmp/bleargh", dtype=float, shape=(100,128), mode='write')
    A[:] = numpy.random.random_integers(0,100,A.shape)
    
    here = socket.gethostname()
    
    ars, revmap = bcast_memmap(dv, 'B', A)
    
    # block to ensure assingment succeeded:
    [ ar.get() for ar in ars ]
    
    
    for datastore_id, targets in revmap.iteritems():
        print datastore_id,
        print rc[targets].apply_sync(lambda : B.__class__.__name__)
        print datastore_id,
        print rc[targets].apply_sync(lambda : numpy.linalg.norm(B,2))
        # print rc[targets].apply_sync(lambda : )
