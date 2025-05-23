import jax.numpy as jnp
import jax

'''
Corpus structure: list of size (n_words, 2)
n_words ALL words in corpus (not just vocabulary)
for each word in corpus, you have (doc_id, word_id)

Z -> array of length n_words: for each corpus[w], Z[w] is the topic index attached to that word
'''
def get_count_matrices(corpus, Z, V, K):
    # assume corpus doc ids from 0 to (m-1)
    m = jnp.max(corpus[:,0]) + 1
    word_topic = jnp.zeros((V,K))
    doc_topic = jnp.zeros((m,K))
    
    def get_count(i, val):
        word_topic, doc_topic = val
        doc_idx, word_idx = corpus[i]
        word_topic = word_topic.at[word_idx, Z[i]].set(word_topic[word_idx, Z[i]] + 1)
        doc_topic = doc_topic.at[doc_idx, Z[i]].set(doc_topic[doc_idx, Z[i]] + 1)
        val = (word_topic, doc_topic)
        return val
        
    val = jax.lax.fori_loop(0, len(corpus), get_count, (word_topic, doc_topic))
    word_topic, doc_topic = val
    dotword_topic = word_topic.sum(axis=0)
    
    return word_topic, doc_topic, dotword_topic


@jax.jit
def sample_L(corpus, Z, word_topic, doc_topic, dotword_topic, alpha, beta, key):
    V = word_topic.shape[0]
    K = word_topic.shape[1]
    
    def g(i, val):
        Z, word_topic, doc_topic, dotword_topic, key = val
        x = corpus[i][1]
        doc_id = corpus[i][0]
        z_old = Z[i]
        # unassign Z[i,j]
        word_topic = word_topic.at[x, z_old].set(word_topic[x,z_old] - 1)
        dotword_topic = dotword_topic.at[z_old].set(dotword_topic[z_old] - 1)
        doc_topic = doc_topic.at[doc_id, z_old].set(doc_topic[doc_id, z_old] - 1)
        # sample
        key, subkey = jax.random.split(key)
        prob = (word_topic[x] + beta) * (doc_topic[doc_id] + alpha) / (dotword_topic + V*beta)
        z = jax.random.choice(key, a=K, p=prob/prob.sum())
        # reassign
        Z = Z.at[i].set(z)
        word_topic = word_topic.at[x, z].set(word_topic[x,z] + 1)
        dotword_topic = dotword_topic.at[z].set(dotword_topic[z] + 1)
        doc_topic = doc_topic.at[doc_id, z].set(doc_topic[doc_id, z] + 1)
        val = Z, word_topic, doc_topic, dotword_topic, key
        return val
        
    
    val = jax.lax.fori_loop(0, len(corpus), g, (Z, word_topic, doc_topic, dotword_topic, key))
    Z, word_topic, doc_topic, dotword_topic, key = val
    val = (corpus, Z, word_topic, doc_topic, dotword_topic, alpha, beta, key)
    return val

def Gibbs_LDA(key, corpus, K, V, alpha, beta, iterations=500, n_samples=100):

    Z = jax.random.choice(key, a=K, shape=(corpus.shape[0],))
    
    word_topic, doc_topic, dotword_topic = get_count_matrices(corpus, Z, V, K)
    
    def sampler(t, val):
        # burnin phase - donot collect samples
        corpus, Z, word_topic, doc_topic, dotword_topic, alpha, beta, key = val
        val = sample_L(corpus, Z, word_topic, doc_topic, dotword_topic, alpha, beta, key)
        corpus, Z, word_topic, doc_topic, dotword_topic, alpha, beta, key = val
        
        return (corpus, Z, word_topic, doc_topic, dotword_topic, alpha, beta, key)
        
    corpus, Z, word_topic, doc_topic, dotword_topic, alpha, beta, key = jax.lax.fori_loop(0, iterations-n_samples, sampler, (corpus, Z, word_topic, doc_topic, dotword_topic, alpha, beta, key))
    
    Beta_samples = jnp.zeros((n_samples, doc_topic.shape[0], doc_topic.shape[1]))
    
    def sampler2(t, val):
        # collect beta hat samples
        corpus, Z, word_topic, doc_topic, dotword_topic, alpha, beta, Beta_samples, key = val
        val = sample_L(corpus, Z, word_topic, doc_topic, dotword_topic, alpha, beta, key)
        corpus, Z, word_topic, doc_topic, dotword_topic, alpha, beta, key = val
        
        beta_hat = doc_topic + alpha
        beta_hat = beta_hat / beta_hat.sum(axis=1)[:,None]
        Beta_samples = Beta_samples.at[t].set(beta_hat)
        
        return (corpus, Z, word_topic, doc_topic, dotword_topic, alpha, beta, Beta_samples, key)
        
    corpus, Z, word_topic, doc_topic, dotword_topic, alpha, beta, Beta_samples, key = jax.lax.fori_loop(0, n_samples, sampler2, (corpus, Z, word_topic, doc_topic, dotword_topic, alpha, beta, Beta_samples, key))
    
    beta_samples = Beta_samples[::5]
    
    return beta_samples.mean(axis=0)