'''
Created by SUNRIT on Feb 3, 2024
Implements: generation of parameters and data for LDA and Tree-LDA
Also implements plotting
Notations:
T - tree
I (#paths), J (depth), K (#nodes) - size of T
K - number of topics
V - vocabulary size (aka ambient dimension)
m - number of documents
n - avg number of words per document
alpha - Dirichlet parameter (scalar)

Corpus structure:
(1) lda_data, tree_lda_data -> (m,n) usual structure
(2) lda_data_list, tree_lda_data_list -> list of length mn (doc_id, word_id)
(3) doc_word -> (m,V) at index [doc_id, word_id] stores frequency

conversion functions:
convert(corpus, from='m_n', to='m_V')
from / to can be 'm_n' (type 1), 'mn' (type 2) or 'm_V' (type 3)
note: for all corpus types 2 and 3 are possible

Remark: For LDA/Tree-LDA Gibbs sampler, need corpus in type 2
'''
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import itertools
import DRT.tree_structure as tree_structure

def generate_topics(K, V, eta, seed):
    key = jax.random.PRNGKey(seed)
    alpha = jnp.ones(V) * eta
    topics = jax.random.dirichlet(key = key, alpha = alpha, shape=(K,))
    topics = topics / topics.sum(axis=1)[:,None]
    return topics

def treelda_data(n_docs, n_words_per_doc, topics, alpha, pi, tree, seed=10):
    # for usual LDA, just use a linear tree
    
    key = jax.random.PRNGKey(seed)
    
    # sizes
    K, V = topics.shape
    I, J, K_ = tree.size
    
    # size checks
    if K != K_:
        raise ValueError("Tree size K does not match topics")
    paths = jnp.array(tree_structure.cpaths(tree), dtype=jnp.int32)
    if len(alpha) != J:
        raise ValueError("Incorrect alpha size: must be J")
    
    
    corpus = jnp.zeros((n_docs, n_words_per_doc), dtype=jnp.int32)
    C = jnp.zeros(n_docs, dtype=jnp.int32)
    L = jnp.zeros((n_docs, n_words_per_doc), dtype=jnp.int32)
    
    # loop within a document
    def fill_doc(j, val):
        document, sub_topics, L_doc, alpha, key = val
        key, subkey = jax.random.split(key)
        x = jax.random.choice(key, a=sub_topics.shape[1], p=sub_topics[L_doc[j]])
        document = document.at[j].set(x)
        return (document, sub_topics, L_doc, alpha, key)
        
        
    # loop across document
    def generate_doc(i, val):
        corpus, C, L, alpha, topics, pi, paths, key = val
        key, subkey = jax.random.split(key)
        c = jax.random.choice(key, a=paths.shape[0], p=pi)
        C = C.at[i].set(c)
        beta = jax.random.dirichlet(key, alpha)
        l = jax.random.choice(key, a=paths.shape[1], p=beta, shape=(n_words_per_doc,))
        
        document = jnp.zeros(n_words_per_doc, dtype=jnp.int32)
        document, _, _, _, key = jax.lax.fori_loop(0, n_words_per_doc, fill_doc, (document, topics[paths[c]], l, alpha, key))
        L = L.at[i].set(l)
        corpus = corpus.at[i].set(document)
        return (corpus, C, L, alpha, topics, pi, paths, key)

    corpus, C, L, _, _, _, _, _ = jax.lax.fori_loop(0, n_docs, generate_doc, (corpus, C, L, alpha, topics, pi, paths, key))
    return corpus, C, L

def convert_list(corpus, V, verbose=True):
    # input is (m,n)
    m = len(corpus)
    words = jnp.zeros((corpus.shape[0]*corpus.shape[1], 2), dtype=jnp.int32)
    def func(i, val):
        corpus, words = val
        
        for j in range(corpus.shape[1]):
            words = words.at[i*corpus.shape[1]+j].set(jnp.array([i, corpus[i,j]], dtype=jnp.int32))
        return corpus, words
        
    _, words = jax.lax.fori_loop(0, m, func, (corpus, words))
    if verbose:
        print(f'Documents: {m}, vocabulary size: {V}, total words = {len(words)}')
    return words

def convert_doc_word_count(corpus, V):
    # converts (m,n) corpur to (m,V) count matrix
    m = corpus.shape[0]
    Y = np.zeros((m, V))
    for i in range(m):
        doc = Counter(corpus[i])
        for v in range(V):
            Y[i,v] = doc[v]
    return Y

def plot_corpus_tree(corpus, V, tree, C=None, topics_true=None, topics_est=None, plot_vocab_simplex=True, title=None, scatter_alpha=0.5, pca_dim1=1, pca_dim2=2):
    corpus = np.array(corpus)
    X = convert_doc_word_count(corpus, V)
    X = X / X.sum(axis=1)[:,None]
    paths = tree_structure.cpaths(tree)
    I, J, K = tree.size
    
    # perform a PCA
    if topics_true is None:
        pca = PCA(n_components=max([pca_dim1, pca_dim2])).fit(X)
    else:
        topics_true = np.array(topics_true)
        pca = PCA(n_components=max([pca_dim1, pca_dim2])).fit(topics_true)
    X_pca = pca.transform(X)[:, tuple([pca_dim1-1, pca_dim2-1])]
    
    if plot_vocab_simplex:
        vocab_boundary = np.eye(V)
        vocab_bd_pca = pca.transform(vocab_boundary)[:, tuple([pca_dim1-1, pca_dim2-1])]
        
    if topics_true is not None:
        topics_true_pca = pca.transform(topics_true)[:, tuple([pca_dim1-1, pca_dim2-1])]
    if topics_est is not None:
        topics_est = np.array(topics_est)
        topics_est_pca = pca.transform(topics_est)[:, tuple([pca_dim1-1, pca_dim2-1])]
        
           
    if topics_true is not None:
        for i in range(I):
            plt.plot(*zip(*itertools.chain.from_iterable(itertools.combinations(topics_true_pca[tuple(paths[i]),],2))), marker='o', linestyle='dotted', label='true simplex '+str(i))
            
    if topics_est is not None:
        for i in range(I):
            plt.plot(*zip(*itertools.chain.from_iterable(itertools.combinations(topics_est_pca[tuple(paths[i]),],2))), color='blue', marker='x', markerfacecolor='blue', linestyle='dotted', label='est simplex '+str(i))
    
    if C is not None:
        for k in np.unique(C):
            idx1 = np.where(C==k)[0]
            plt.scatter(X_pca[idx1,0], X_pca[idx1,1], label="C="+str(k), alpha=scatter_alpha)
    else:
        plt.scatter(X_pca[:,0], X_pca[:,1], color='gray', alpha=scatter_alpha)
    plt.xlabel('PCA'+str(pca_dim1))
    plt.ylabel('PCA'+str(pca_dim2))
    if title is not None:
        plt.title(str(title))
    plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
def plot(ax, X, V, tree, C=None, topics_true=None, topics_est=None, title=None, scatter_alpha=0.5, pca_dim1=1, pca_dim2=2, legend=True):
    paths = tree_structure.cpaths(tree)
    I, J, K = tree.size
    
    # perform a PCA
    if topics_true is None:
        pca = PCA(n_components=max([pca_dim1, pca_dim2])).fit(X)
    else:
        topics_true = np.array(topics_true)
        pca = PCA(n_components=max([pca_dim1, pca_dim2])).fit(topics_true)
    X_pca = pca.transform(X)[:, tuple([pca_dim1-1, pca_dim2-1])]
        
    if topics_true is not None:
        topics_true_pca = pca.transform(topics_true)[:, tuple([pca_dim1-1, pca_dim2-1])]
    if topics_est is not None:
        topics_est = np.array(topics_est)
        topics_est_pca = pca.transform(topics_est)[:, tuple([pca_dim1-1, pca_dim2-1])]
        
    ax = ax or plt.gca()
    if topics_true is not None:
        for i in range(I):
            line, = ax.plot(*zip(*itertools.chain.from_iterable(itertools.combinations(topics_true_pca[tuple(paths[i]),],2))), marker='o', linestyle='dotted', label='true simplex '+str(i))
            
    if topics_est is not None:
        for i in range(I):
            line, = ax.plot(*zip(*itertools.chain.from_iterable(itertools.combinations(topics_est_pca[tuple(paths[i]),],2))), color='blue', marker='x', markerfacecolor='blue', linestyle='dotted', label='est simplex '+str(i))
    
    if C is not None:
        for k in np.unique(C):
            idx1 = np.where(C==k)[0]
            line = ax.scatter(X_pca[idx1,0], X_pca[idx1,1], label="C="+str(k), alpha=scatter_alpha)
    else:
        line = ax.scatter(X_pca[:,0], X_pca[:,1], color='gray', alpha=scatter_alpha)
    ax.set_xlabel('PCA'+str(pca_dim1))
    ax.set_ylabel('PCA'+str(pca_dim2))
    if title is not None:
        ax.set_title(str(title))
    if legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return line
    
def expanded_topics(topics, paths):
    # gives a 3d topic array: [path, depth, word] (I, J, V)
    I, J = paths.shape
    V = topics.shape[1]
    topics_expanded = np.zeros((I, J, V))
    for i in range(I):
        topics_expanded[i] = topics[tuple(paths[i]),]
    return topics_expanded