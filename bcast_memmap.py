from IPython import parallel

@parallel.util.interactive
def reconstruct_memmap(name, fname, dtype, shape):
    import numpy
    X = numpy.memmap(fname, shape=shape, dtype=dtype, mode='copyonwrite')
    globals().update({name:X})

# this fix is pending in master (0.13)
@parallel.util.interactive
def _push(**ns):
    globals().update(ns)

def bcast_memmap(view, name, X):
    import socket
    
    hostmap = dv.apply_async(socket.gethostname).get_dict()
    here = socket.gethostname()
    local_engines = [ eid for eid in hostmap if hostmap[eid] == here ]
    remote_engines = [ eid for eid in hostmap if hostmap[eid] != here ]
    
    # # fake with even/odd while testing local
    # local_engines = dv.client.ids[::2]
    # remote_engines = dv.client.ids[1::2]
    
    ar = ar2 = None
    
    if local_engines:
        local_view = dv.client.direct_view(local_engines)
        ar = local_view.apply(reconstruct_memmap, name, X.filename, X.dtype, X.shape)
    
    if remote_engines:
        remote_view = dv.client.direct_view(remote_engines)
        # this can just be push after fix
        ar2 = remote_view.apply(_push, **{name : X})
    
    return ar, ar2

if __name__ == '__main__':
    import socket
    import numpy
    
    
    A = numpy.memmap("/tmp/bleargh", dtype=float, shape=(100,128), mode='write')
    rc = parallel.Client()
    dv = rc[:]
    
    ar, ar2 = bcast_memmap(dv, 'B', A)
    
    # block to ensure assingment succeeded:
    if ar:
        ar.get()
    if ar2:
        ar2.get()
    
    hostmap = dv.apply_async(socket.gethostname).get_dict()
    here = socket.gethostname()
    
    types = dv.apply_async(lambda : type(B)).get_dict()
    print types
    for eid in types:
        if hostmap[eid] == here:
            assert types[eid] is numpy.memmap, "local engine got wrong type: %r" % types[eid]
        else:
            assert types[eid] is numpy.ndarray, "remote engine got wrong type: %r" % types[eid]
