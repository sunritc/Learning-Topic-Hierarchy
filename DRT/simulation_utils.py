'''
Use multiple initializations for each case

A case is defined by the tuple (n, m, seed)
where m is #docs, n is #words/doc, seed is for data

Initializations labels:
1 - long/trunc alpha=0.1, iters=700 (initialize_trunc)
2 - long/trunc alpha=0.3, iters=700
3 - long/trunc alpha=0.5, iters=700
4 - long/trunc alpha=0.7, iters=700
5 - old/subset alpha=0.1, iters=700 (initialize_subset) 
6 - old/subset alpha=0.3, iters=700
7 - old/subset alpha=0.5, iters=700
8 - old/subset alpha=0.7, iters=700
'''

from DRT.tree_structure import *
from sklearn.decomposition import PCA
import itertools
from DRT.initializers import *
from DRT.sampler_DRT import Gibbs_TreeLDA
from time import time
from DRT.metrics import *
from jax.scipy.special import logsumexp
from joblib import Parallel, delayed

'''
plot and fit are generic functions (can be used with any dataset)

fit_tree1 and fit_tree2 are tailored to the simulation setup
'''

def fit(key, corpus, V, alpha0, tree, n_jobs=-1, eta=1.0, iterations=2500, n_samples=500, random_inits=False):
    '''
    corpus is (N,2) -> first column is doc_id, 2nd column is words
    '''
    m = len(np.unique(corpus[:,0]))
    jax_corpus = jnp.array(corpus, dtype=jnp.int32)
    
    def chains(i, corpus=jax_corpus, alpha=alpha0, key=key, tree=tree, random_inits=random_inits):
        paths = jnp.array(cpaths(tree), dtype=jnp.int32)
        
        K = tree.size[2]
        I, J = paths.shape
        if random_inits:
            np.random.seed(100*i)
            c_init = jnp.array(np.random.choice(I, size=(len(corpus),)), dtype=jnp.int32)
        else:
            if i == 0:
                c_init = initialize_trunc(key, corpus, tree, K, V, alpha=0.1, iters=700)
            elif i == 1:
                c_init = initialize_trunc(key, corpus, tree, K, V, alpha=0.3, iters=700)
            elif i == 2:
                c_init = initialize_trunc(key, corpus, tree, K, V, alpha=0.5, iters=700)
            elif i == 3:
                c_init = initialize_trunc(key, corpus, tree, K, V, alpha=0.7, iters=700)
            elif i == 4:
                c_init = initialize_subset(key, corpus, tree, K, V, alpha=0.1, iters=700)
            elif i == 5:
                c_init = initialize_subset(key, corpus, tree, K, V, alpha=0.3, iters=700)
            elif i == 6:
                c_init = initialize_subset(key, corpus, tree, K, V, alpha=0.5, iters=700)
            elif i == 7:
                c_init = initialize_subset(key, corpus, tree, K, V, alpha=0.7, iters=700)
        
        time_start = time()
        Theta_samples, Pi_samples, loglik_terms = Gibbs_TreeLDA(key, corpus, K, V, paths, alpha=alpha0, eta=eta, pi0=1.0, iterations=iterations, n_samples=n_samples, C_init=c_init)
        time_end = time()
        Theta_samples = Theta_samples[::20]
        Pi_samples = Pi_samples[::20]
        loglik_terms = loglik_terms[::20]
        
        loglik = (logsumexp(-loglik_terms) - jnp.log(len(loglik_terms)))/m
        
        return (Theta_samples, Pi_samples, loglik.item(), time_end-time_start)
        
    results =  Parallel(n_jobs=n_jobs)(delayed(chains)(i) for i in range(8))
    
    logliks = np.array([result[2] for result in results])
    idx = np.argmin(logliks)
    return results[idx] # (Theta_samples, pi_samples, loglik, time)

def fit_simul(corpus, K, V, topics, pi, alpha0, key, tree_number = 1):
    
    if tree_number == 1:
        tree = Tree({0: [1,2], 1: [3], 2: [4]})
        sharing_metric = correct_structure_tree1
    elif tree_number == 2:
        tree = Tree({0: [1,2], 1: [3,4], 2: [5,6]})
        sharing_metric = correct_structure
    else:
        raise ValueError('tree number either 1 or 2 for simulations')
    Theta_samples, Pi_samples, negloglik, time = fit(key, corpus, V, alpha0, tree, n_jobs=-1, iterations=5000, n_samples=500, random_inits=False)
    
    paths = jnp.array(cpaths(tree), dtype=jnp.int32)
    distances = parallel_metrics2(topics, Theta_samples, pi, Pi_samples, paths).mean(axis=0)
    correct = sharing_metric(topics, Theta_samples[-1], tree)
    err_W1 = distances[:,0]
    err_L2 = distances[:,1]
    return err_W1, err_L2, correct, negloglik, time