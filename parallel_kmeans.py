#
#
#
#
#
import sys

import numpy as np

from IPython.parallel.util import interactive
from IPython import parallel

def print_flush(msg):
    print msg
    sys.stdout.flush()

@interactive
def load_data(name, partition_id, n_partitions):
    """load partition of data into global var `name`"""
    from sklearn.datasets import fetch_20newsgroups_vectorized
    from sklearn.utils import gen_even_slices
    dataset = fetch_20newsgroups_vectorized('test')
    size = dataset.data.shape[0]
    slices = list(gen_even_slices(size, n_partitions))
    
    part = dataset.data[slices[partition_id]]
    # put it in globals
    globals().update({name : part})
    return part.shape

def init_kmeans(data, n_clusters, batch_size):
    
    from sklearn.cluster import MiniBatchKMeans
    
    kmeans = MiniBatchKMeans(n_clusters, batch_size=batch_size).partial_fit(data[:batch_size])
    
    return kmeans

def compute_kmeans(kmeans, data):
    
    import numpy as np
    
    x = np.random.random_integers(0, data.shape[0]-1, kmeans.batch_size)
    
    kmeans.partial_fit(data[x])
    
    return kmeans

def run_kmeans(view, name, n_clusters, loader, batch_size=100):
    
    ids = view.targets or view.client.ids
    n_partitions = len(ids)
    
    rdata = parallel.Reference(name)
    
    view.scatter('partition_id', range(len(ids)), flatten=True)
    partition_id = parallel.Reference('partition_id')
    
    print_flush("Loading Data")
    load_ar = view.apply_async(loader, name, partition_id, n_partitions)
    
    # DEBUG: block on load
    total_size = sum([ shape[0] for shape in load_ar ])
    partition_size = max([ shape[0] for shape in load_ar ])
    _, nfeatures = load_ar[0]
    
    e0 = view.client[ids[0]]
    kmeans_ar = e0.apply_async(init_kmeans, rdata, n_clusters, batch_size)

    # DEBUG: block on init
    kmeans_ar.get()
    
    # FIXME: TEMPORARILY TERRIBLE BROADCAST
    print_flush("Broadcasting initialized kmeans...")
    view['kmeans'] = kmeans_ar.get()
    print_flush("done.")
    
    rkmeans = parallel.Reference('kmeans')
    print_flush("nsteps %s" % (partition_size // batch_size))
    for step in range(partition_size // batch_size):
        print_flush(step)
        ar = view.apply_async(compute_kmeans, rkmeans, rdata)
        
        # FIXME: GLOBAL COMM VIA CLIENT
        # kmeans = ar.get()
        mean_centers = np.zeros((n_clusters, nfeatures))
        for km in ar:
            mean_centers += km.cluster_centers_
        mean_centers /= len(ar.msg_ids)
        
        # FIXME: GLOBAL
        def new_centers(km, centers):
            
            km.cluster_centers_ = 1*centers
        ar = view.apply_async(new_centers, rkmeans, mean_centers)
    
    km.cluster_centers_ = mean_centers
    
    return km

