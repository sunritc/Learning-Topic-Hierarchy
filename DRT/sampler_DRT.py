import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp
import jax
from jax import jit, vmap
from functools import partial
from DRT.tree_structure import *
from jax.scipy.stats import multinomial as mult


'''
# New modifications based on overleaf latest algorithm

# Runs for burnin without collecting samples
# then runs collecting samples
# CURRENTLY: takes burnin as parameter

Corpus structure: list of size (n_words, 2)
n_words ALL words in corpus (not just vocabulary)
for each word in corpus, you have (doc_id, word_id)

C -> array of length m: C[i] is path label for document i
L -> array of length n_words: for each corpus[w], L[w] is the depth label

notation: z = topic id given by z[w] = paths[C[corpus[w,0]], L[w]] for each word w in corpus

Count matrices:
(1) doc_word_depth  (m, V, J)
(2) word_topic      (V, K)
(3) doc_topic       (m, K)
(4) dotword_topic   (K,)
(5) doc_word        (m, V) -> static
(6) path_counts     (I,)

Hyperparameters -> pi0, alpha, eta
pi0 (prior pi), eta (prior theta), alpha (prior beta) 

'''


def get_count_matrices(corpus, K, V, paths, C, L):
    m = jnp.max(corpus[:,0]) + 1
    
    doc_word_depth = jnp.zeros((m,V,paths.shape[1]))
    word_topic = jnp.zeros((V,K))
    doc_topic = jnp.zeros((m,K))
    path_counts = jnp.zeros(paths.shape[0])
    for i in range(paths.shape[0]):
        new_c = jnp.where(C==i, 1, 0)
        path_counts = path_counts.at[i].set(new_c.sum())
    
    def get_count(i, val):
        doc_word_depth, word_topic, doc_topic, path_counts = val
        doc_idx, word_idx = corpus[i]
        z = paths[C[doc_idx], L[i]]
        doc_word_depth = doc_word_depth.at[doc_idx, word_idx, L[i]].set(doc_word_depth[doc_idx, word_idx, L[i]] + 1)
        word_topic = word_topic.at[word_idx, z].set(word_topic[word_idx, z] + 1)
        doc_topic = doc_topic.at[doc_idx, z].set(doc_topic[doc_idx, z] + 1)
        val = (doc_word_depth, word_topic, doc_topic, path_counts)
        return val
        
    val = jax.lax.fori_loop(0, len(corpus), get_count, (doc_word_depth, word_topic, doc_topic, path_counts))
    doc_word_depth, word_topic, doc_topic, path_counts = val
    
    dotword_topic = word_topic.sum(axis=0)
    doc_word = doc_word_depth.sum(axis=2)
    
    return doc_word_depth, word_topic, doc_topic, dotword_topic, doc_word, path_counts

@jax.jit
def sample_L(corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, paths, alpha, eta, pi0, key):
    V = word_topic.shape[0]
    J = paths.shape[1]
    
    def func_L(i, val):
        L, doc_word_depth, word_topic, doc_topic, dotword_topic, key = val
        x = corpus[i][1]
        doc_id = corpus[i][0]
        c_old = C[doc_id]
        l_old = L[i]
        z_old = paths[c_old, l_old]
        # unassign Z[i,j]
        word_topic = word_topic.at[x, z_old].set(word_topic[x,z_old] - 1)
        dotword_topic = dotword_topic.at[z_old].set(dotword_topic[z_old] - 1)
        doc_topic = doc_topic.at[doc_id, z_old].set(doc_topic[doc_id, z_old] - 1)
        doc_word_depth = doc_word_depth.at[doc_id, x, l_old].set(doc_word_depth[doc_id, x, l_old] - 1)
        # sample
        topic_idx = paths[c_old]
        key, subkey = jax.random.split(key)
        prob = (word_topic[x, topic_idx] + eta) * (doc_topic[doc_id, topic_idx] + alpha) / (dotword_topic[topic_idx] + V*eta)
        l_new = jax.random.choice(key, a=J, p=prob/prob.sum())
        # reassign
        L = L.at[i].set(l_new)
        z = paths[c_old, l_new]
        word_topic = word_topic.at[x, z].set(word_topic[x,z] + 1)
        dotword_topic = dotword_topic.at[z].set(dotword_topic[z] + 1)
        doc_topic = doc_topic.at[doc_id, z].set(doc_topic[doc_id, z] + 1)
        doc_word_depth = doc_word_depth.at[doc_id, x, l_new].set(doc_word_depth[doc_id, x, l_new] + 1)
        val = (L, doc_word_depth, word_topic, doc_topic, dotword_topic, key)
        return val
        
    
    L, doc_word_depth, word_topic, doc_topic, dotword_topic, key = jax.lax.fori_loop(0, len(corpus), func_L, (L, doc_word_depth, word_topic, doc_topic, dotword_topic, key))
    
    return (corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, paths, alpha, eta, pi0, key)
    
@jax.jit
def sample_C(corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, paths, eta, pi0, key):
    V = word_topic.shape[0]
    J = paths.shape[1]
    
    def func_C(w, val):
        C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, key = val
        c_old = C[w]
        # step 1 - calculate document-specific word_topic count matrix and the first term
        doc_specific_word_topic = jnp.zeros(word_topic.shape)
        doc_specific_word_topic = doc_specific_word_topic.at[:,paths[c_old]].set(doc_word_depth[w])
        B_temp = word_topic - doc_specific_word_topic + eta # (V,K)
        term1 = jnp.sum(gammaln(B_temp.sum(axis=0))) - jnp.sum(gammaln(B_temp)) # withut the current document
        path_counts = path_counts.at[c_old].set(path_counts[c_old] - 1)
        # step 2 - calculate probability vector
        prob = jnp.zeros(paths.shape[0])
        for i in jnp.arange(paths.shape[0], dtype=jnp.int32):
            permuted_doc_specific_word_topic = jnp.zeros((word_topic.shape[0], word_topic.shape[1]))
            permuted_doc_specific_word_topic = permuted_doc_specific_word_topic.at[:,paths[i]].set(doc_specific_word_topic[:,paths[c_old]])
            proposed_word_topic = B_temp + permuted_doc_specific_word_topic
            prob = prob.at[i].set(term1 + jnp.sum(gammaln(proposed_word_topic)) - jnp.sum(gammaln(proposed_word_topic.sum(axis=0))) + jnp.log(pi0 + path_counts[i]))
        prob = jnp.exp(prob - logsumexp(prob))
        key, subkey = jax.random.split(key)
        
        # step 3 - draw from the distribution for C[w]
        c = jax.random.choice(key, a=paths.shape[0], p=prob)
        
        # step 4 - reassign all objects
        C = C.at[w].set(c)
        permuted_doc_specific_word_topic = jnp.zeros((word_topic.shape[0], word_topic.shape[1]))
        permuted_doc_specific_word_topic = permuted_doc_specific_word_topic.at[:,paths[c]].set(doc_specific_word_topic[:,paths[c_old]])
        proposed_word_topic = B_temp + permuted_doc_specific_word_topic
        
        word_topic = word_topic - doc_specific_word_topic + permuted_doc_specific_word_topic
        dotword_topic = word_topic.sum(axis=0)
        doc_topic = doc_topic.at[w].set(doc_topic[w] - doc_specific_word_topic.sum(axis=0) + permuted_doc_specific_word_topic.sum(axis=0))
        # return 
        path_counts = path_counts.at[c].set(path_counts[c] + 1)
        val = C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, key
        return val
        
    
    C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, key = jax.lax.fori_loop(0, doc_topic.shape[0], func_C, (C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, key))
    
    return (corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, paths, eta, pi0, key)


def Gibbs_TreeLDA(key, corpus, K, V, paths, alpha, eta=1.0, pi0=1.0, iterations=1000, n_samples=500, C_init=None, L_init=None):
    
    I, J = paths.shape
    m = jnp.max(corpus[:,0]) + 1
    if C_init is None:
        C = jax.random.choice(key, a=I, shape=(m,)).astype(jnp.int32)
    else:
        C = jnp.array(C_init, dtype=jnp.int32)
    if L_init is None:
        L = jax.random.choice(key, a=J, shape=(corpus.shape[0],)).astype(jnp.int32)
    else:
        L = jnp.array(L_init, dtype=jnp.int32)
        
    doc_word_depth, word_topic, doc_topic, dotword_topic, doc_word, path_counts = get_count_matrices(corpus, K, V, paths, C, L)
        
    burnin = iterations - n_samples
    
    def sampler1(t, val):
        corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, paths, alpha, eta, pi0, key = val
        corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, paths, alpha, eta, pi0, key = sample_L(corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, paths, alpha, eta, pi0, key)
        corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, paths, eta, pi0, key = sample_C(corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, paths, eta, pi0, key)
        return (corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, paths, alpha, eta, pi0, key)
        
    corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, paths, alpha, eta, pi0, key = jax.lax.fori_loop(0, burnin, sampler1, (corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, paths, alpha, eta, pi0, key))
    
    Theta_samples = jnp.zeros((n_samples, K, V))
    Pi_samples = jnp.zeros((n_samples, I))
    loglik_terms = jnp.zeros(n_samples)
    
    def sampler2(t, val):
        Theta_samples, Pi_samples, loglik_terms, corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, paths, alpha, eta, pi0, key = val
        corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, paths, alpha, eta, pi0, key = sample_L(corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, paths, alpha, eta, pi0, key)
        corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, paths, eta, pi0, key = sample_C(corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, paths, eta, pi0, key)
        
        theta_hat = word_topic.T + eta
        theta_hat = theta_hat / theta_hat.sum(axis=1)[:,None]
        
        pi_hat = path_counts + pi0
        pi_hat = pi_hat / pi_hat.sum()
    
        Theta_samples = Theta_samples.at[t].set(theta_hat)
        Pi_samples = Pi_samples.at[t].set(pi_hat)
        
        loglik_term = K * gammaln(V * eta) - K * V * gammaln(eta)
        temp = word_topic + eta
        loglik_term = loglik_term + jnp.sum(gammaln(temp)) - jnp.sum(gammaln(temp.sum(axis=0)))
        loglik_terms = loglik_terms.at[t].set(loglik_term)
        
        return (Theta_samples, Pi_samples, loglik_terms, corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, paths, alpha, eta, pi0, key)
    
    Theta_samples, Pi_samples, loglik_terms, corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, paths, alpha, eta, pi0, key = jax.lax.fori_loop(0, n_samples, sampler2, (Theta_samples, Pi_samples, loglik_terms, corpus, L, C, doc_word_depth, word_topic, doc_topic, dotword_topic, path_counts, paths, alpha, eta, pi0, key))
    
    return Theta_samples, Pi_samples, loglik_terms


def expanded_topics(topics, paths):
    # gives a 3d topic array: [path, depth, word] (I, J, V)
    I, J = paths.shape
    V = topics.shape[1]
    topics_expanded = np.zeros((I, J, V))
    for i in range(I):
        topics_expanded[i] = topics[tuple(paths[i]),]
    return topics_expanded

def loglikelihood(X, topics, pi, alpha, paths, L=1000):
    '''
    X is (m,V) counts
    '''
    Topics = expanded_topics(topics, paths) # (I, J, V)
    m, V = X.shape
    I, J = paths.shape
    beta = np.random.dirichlet(alpha=alpha, size=L)
    Eta = jnp.zeros((I, L, V))
    for i in range(I):
        Eta = Eta.at[i].set(beta @ Topics[i])
    
    f = lambda x, p: mult.logpmf(x, x.sum(), p)
    ff = vmap(f, (None, 0), (0))
    
    def g(x, pi, eta):
        # x -> V, pi -> I, eta -> (I,V)
        return jnp.log(pi) + ff(x, eta) # len I
    
    gg = vmap(g, (None, None, 1), (1)) # gg(x, pi, Eta) -> (I,L)
    ggg = vmap(gg, (0, None, None), (0)) # (m,I,L)
    A = ggg(X, pi, Eta) # (m,I,L)
    A_ = logsumexp(A, axis=(1,2)) # (m,)
    return A_.mean().item()