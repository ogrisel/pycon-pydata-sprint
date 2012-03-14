"""
Optimized broadcast for numpy arrays

Data is sent from the client to each 'datastore' at most once,
and loaded into memmapped arrays.

In this example, the 'datastore' is the '/tmp' directory on each physical
machine.

General flow:

0. hash data
1. query all engines for datastore ID (default: hostname),
   remote filename as function of hash, and whether data
   is already present on the machine.
2. foreach datastore *not* local to the Client, which has not yet seen the data:
        * send data to one engine with access to the datastore
        * store in file for memmap loading
3. on *all* engines, load data as memmapped file from datastore

"""


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

def default_identify(checksum):
    """default datastore-identification function.
<<<<<<< HEAD
    
    Identifies 'datastores' by hostname, and stores data files
    in /tmp.
    
    Parameters
    ----------
    
    checksum : str
        The md5 hash of the array.  Cached filenames should be
        a function of this.
    
    Returns
    -------
    
=======

    Identifies 'datastores' by hostname, and stores data files
    in /tmp.

    Parameters
    ----------

    checksum : str
        The md5 hash of the array.  Cached filenames should be
        a function of this.

    Returns
    -------

>>>>>>> origin/master
    datastore_id : str
        currently socket.gethostname().  This should identify a
        locality, wrt. the data storage mechanism.  Data is only
        transferred to one engine for each datastore_id.
<<<<<<< HEAD
    
=======

>>>>>>> origin/master
    filename : path
        The filename (or url, object id, etc.) for this data to
        be stored in.  This should be a unique function of the
        checksum.
<<<<<<< HEAD
    
=======

>>>>>>> origin/master
    exists : bool
        Whether the data is already available in the datastore.
        """
    import os,socket
    fname = os.path.join("/tmp", checksum)
    return socket.gethostname(), fname, os.path.exists(fname)

<<<<<<< HEAD
=======

>>>>>>> origin/master
def save_for_memmap(A, fname):
    """save array A as memmapped array"""
    import os.path
    import numpy
<<<<<<< HEAD
    
    if os.path.exists(fname):
        # file already exists, nothing to do
        return
    
=======

    if os.path.exists(fname):
        # file already exists, nothing to do
        return

>>>>>>> origin/master
    mmA = numpy.memmap(fname, mode='w+', dtype=A.dtype, shape=A.shape)
    mmA[:] = A
    # maybe need to flush?
    # mmA.flush()

def datastore_mapping(view, identify_f, checksum):
    """generate various mappings of datastores and paths"""
    mapping = dv.apply_async(identify_f, checksum).get_dict()
    here, _, __ = identify_f(checksum)
<<<<<<< HEAD
    
=======

>>>>>>> origin/master
    # reverse mapping, so we have a list of engine IDs per datastore
    revmap = defaultdict(list)
    paths = {}
    for eid, (datastore_id, path, exists) in mapping.iteritems():
        revmap[datastore_id].append(eid)
        paths[datastore_id] = (path, exists)
<<<<<<< HEAD
    
=======

>>>>>>> origin/master
    return here, revmap, paths

def bcast_memmap(view, name, X, identify_f=default_identify):
    """broadcast X as memmapped arrays on all engines in the view
<<<<<<< HEAD
    
    Ultimate result: a memmapped array with the contents of X
    will be stored in globals()[name] on each engine of the view.
    
    Efforts are made to minimize network traffic:
    
    * only send to one engine per datastore (host)
    * only send if data is not already present in each store
    
    broadcast X
    """
    client = view.client
    
    # checksum array for filename
    checksum = md5(X).hexdigest()
    
=======

    Ultimate result: a memmapped array with the contents of X
    will be stored in globals()[name] on each engine of the view.

    Efforts are made to minimize network traffic:

    * only send to one engine per datastore (host)
    * only send if data is not already present in each store

    broadcast X
    """
    client = view.client

    # checksum array for filename
    checksum = md5(X).hexdigest()

>>>>>>> origin/master
    here, revmap, paths = datastore_mapping(view, identify_f, checksum)

    ars = []
    mm_ars = []
<<<<<<< HEAD
    
=======

>>>>>>> origin/master
    # perform push to first engine of each non-local datastore:
    for datastore_id, targets in revmap.iteritems():
        if datastore_id != here:
            fname, exists = paths[datastore_id]
            # if file exists, nothing to do this round
            if exists:
                print "nothing to send to", datastore_id
                continue
<<<<<<< HEAD
            
=======

>>>>>>> origin/master
            print "sending data to", datastore_id
            # push to first target at datastore
            e0 = client[targets[0]]
            # ar = e0.apply(_push, **{name : X})
            # # DEBUG:
            # # ar.get()
            # ars.append(ar)
<<<<<<< HEAD
            
=======

>>>>>>> origin/master
            # save to file for memmapping on other engines
            ar = e0.apply_async(save_for_memmap, X, fname)
            ars.append(ar)
            mm_ars.append(ar)
<<<<<<< HEAD
    
=======

>>>>>>> origin/master
    # wait on pushes, to ensure files are ready for memmap in next step
    # this could be done without blocking the client with a barrier
    # (MPI or ZMQ or file lock, etc.)
    [ ar.get() for ar in mm_ars ]
<<<<<<< HEAD
    
=======

>>>>>>> origin/master
    # loop through datastores
    for datastore_id, targets in revmap.iteritems():
        if datastore_id != here:
            fname, exists = paths[datastore_id]
<<<<<<< HEAD
            
=======

>>>>>>> origin/master
            # load from memmapped file on engines after the first for this datastore
            other = client[targets]
            ar = other.apply_async(load_memmap, name, fname, X.dtype, X.shape)
            # DEBUG:
            # ar.get()
            ars.append(ar)
<<<<<<< HEAD
            
=======

>>>>>>> origin/master
        else:
            if not isinstance(X, numpy.memmap):
                fname, exists = paths[datastore_id]
                if not exists:
                    save_for_memmap(X, fname)
            else:
                fname = X.filename
            # local engines, load from original memmapped file
            ar = client[targets].apply_async(load_memmap, name, fname, X.dtype, X.shape)
            # DEBUG:
            # ar.get()
            ars.append(ar)
<<<<<<< HEAD
            
=======

>>>>>>> origin/master
    return ars, revmap

if __name__ == '__main__':
    import socket
<<<<<<< HEAD
    import sys
    rc = parallel.Client(profile=sys.argv[1])
=======

    rc = parallel.Client(profile='vm')
>>>>>>> origin/master
    dv = rc[:]
    with dv.sync_imports():
        import numpy
        from hashlib import md5

    A = numpy.memmap("/tmp/bleargh", dtype=float, shape=(100,128), mode='write')
    numpy.random.seed(10)
    # A = numpy.empty((100,128))
    A[:] = numpy.random.random_integers(0,100,A.shape)
<<<<<<< HEAD
    
    here = socket.gethostname()
    
    ars, revmap = bcast_memmap(dv, 'B', A)
    
    # block here to raise any potential exceptions:
    [ ar.get() for ar in ars ]
    
    
=======

    here = socket.gethostname()

    ars, revmap = bcast_memmap(dv, 'B', A)

    # block here to raise any potential exceptions:
    [ ar.get() for ar in ars ]


>>>>>>> origin/master
    for datastore_id, targets in revmap.iteritems():
        print datastore_id,
        print rc[targets].apply_sync(lambda : B.filename[:12])
        print datastore_id,
        print rc[targets].apply_sync(lambda : md5(B).hexdigest()[:7])
        # print rc[targets].apply_sync(lambda : )
