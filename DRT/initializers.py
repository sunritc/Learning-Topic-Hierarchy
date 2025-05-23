import jax.numpy as jnp
import jax
from DRT.sampler_LDA import Gibbs_LDA
from DRT.tree_structure import *
from sklearn.cluster import KMeans, AgglomerativeClustering
from copy import deepcopy
from itertools import permutations
from collections import Counter

class document_reader():
    def __init__(self):
        self.n_docs = 0
        
    def process(self, corpus):
        corpus = np.array(corpus, dtype=np.int32)
        doc_id_dict = {}
        word_id_dict = {}
        doc_counter, word_counter = 0, 0
        new_corpus = np.zeros(corpus.shape, dtype=np.int32)
        m = len(np.unique(corpus[:,0]))
        V = len(np.unique(corpus[:,1]))
        doc_lens = np.zeros(m)
        word_counts = np.zeros(V)
        
        for i, (doc_id, word_id) in enumerate(corpus):
            if doc_id not in doc_id_dict:
                doc_id_dict[doc_id] = doc_counter
                doc_counter += 1
            if word_id not in word_id_dict:
                word_id_dict[word_id] = word_counter
                word_counter += 1
            new_doc_id = doc_id_dict[doc_id]
            new_word_id = word_id_dict[word_id]
            new_corpus[i] = [new_doc_id, new_word_id]
            doc_lens[new_doc_id] += 1
            word_counts[new_word_id] += 1
        self.corpus = jnp.array(new_corpus, dtype=jnp.int32)
        self.doc_id_dict = doc_id_dict
        self.word_id_dict = word_id_dict
        self.doc_sizes = doc_lens
        self.word_counts = word_counts
        self.n_docs = len(doc_id_dict)
        self.n_vocab = len(word_id_dict)
        assert m == self.n_docs
        assert V == self.n_vocab
        
    def get_docs_by_id(self, ids, id_type='old'):
        # ids refer to old id, as in original fed corpus
        corpus_subset = None
        for id in ids:
            if id_type == 'old':
                new_id = self.doc_id_dict[id]
            elif id_type == 'new':
                new_id = id
            doc = self.corpus[self.corpus[:,0] == new_id]
            if corpus_subset is None:
                corpus_subset = doc
            else:
                corpus_subset = np.append(corpus_subset, doc, axis=0)
        return jnp.array(corpus_subset, dtype=jnp.int32)
    
def subset_docs(indices, node, contains=1):
    # selects rows of indices whose (node)th entry is contain
    # if contains = 1: selects subset which contains given node
    # if contains = 0: selects subset which does not contain given node
    # indices is beta_hat_subset from get_LDA_results function
    sub_indx = []
    for i in range(indices.shape[0]):
        if indices[i, node] == contains:
            sub_indx.append(i)
    return indices[tuple(sub_indx),:]

def find_children(indices, n_chil, parent_history, except_nodes):
    # except_nodes may be those from a different path
    # pick docs containing parent_history
    sub_indices = indices.copy()
    sub_indices = np.array(sub_indices)
    for node in parent_history:
        sub_indices = subset_docs(sub_indices, node, 1)
    sums = sub_indices.sum(axis=0)
    # pick the top n_chil without those in parent_history and except_nodes
    for node in parent_history:
        sums[node] = -1
    for node in except_nodes:
        sums[node] = -1
    return jnp.argpartition(sums, -n_chil)[-n_chil:]


def get_LDA_results(key, corpus, J, K, V, alpha=0.05, iterations=500):
    beta_hat = Gibbs_LDA(key, corpus, K, V, 1.0, alpha, iterations, 100)
    idx = jnp.argpartition(beta_hat, -J)
    idx = idx[:, -J:]
    
    beta_hat_subset = jnp.zeros(beta_hat.shape)
    beta_hat_trunc = jnp.zeros(beta_hat.shape)
    
    def func(i, val):
        beta_hat_trunc, beta_hat_subset, idx = val
        
        vec = jnp.zeros(beta_hat_subset.shape[1])
        vec = vec.at[idx[i]].set(1)
        beta_hat_subset = beta_hat_subset.at[i].set(vec)
        
        vec = vec.at[idx[i]].set(beta_hat[i, idx[i]])
        beta_hat_trunc = beta_hat_trunc.at[i].set(vec / vec.sum())
        
        return (beta_hat_trunc, beta_hat_subset, idx)
    
    val = jax.lax.fori_loop(0, beta_hat.shape[0], func, (beta_hat_trunc, beta_hat_subset, idx))
    beta_hat_trunc, beta_hat_subset, idx = val
    
    return beta_hat_trunc, beta_hat_subset, beta_hat

def initialize1(beta_hat_trunc, tree):
    '''
    can pass beta_hat_trunc or beta_hat_subset
    metric can be euclidean or manhattan
    use the trunc one for the new method (Long) - clustering based on number of children
    '''
    node = 0 # start at root
    idx = np.zeros(len(beta_hat_trunc))
    linkage = 'ward'
    
    def node_cluster(node, X, idx):
        idx_node = np.where(idx == node)[0]
        X_sub = X[idx_node]
        if node not in tree.leaves_():
            clustering = AgglomerativeClustering(n_clusters=len(tree.children_(node)), metric="euclidean", linkage=linkage).fit(X_sub)
            labels = clustering.labels_
            labels_new = np.zeros(len(labels))
            for i, k in enumerate(tree.children_(node)):
                labels_new[labels == i] = k
            idx[idx_node] = labels_new
        return idx
    
    for node in tree.nodes:
        idx = node_cluster(node, beta_hat_trunc, idx)
        
    # returns label in [I]
    labels = np.zeros(len(beta_hat_trunc))
    for i, k in enumerate(tree.leaves_()):
        labels[idx == k] = i
    return labels.astype(np.int32)

def initialize2(beta_hat_subset, beta_hat):
    # onlt for branching=2, depth=3
    K = beta_hat.shape[1]
    root = beta_hat_subset.sum(axis=0).argmax()
    level1 = find_children(beta_hat_subset, 2, [root], [])
    two_leaves = find_children(beta_hat_subset, 2, [root, level1[0]], [level1[1]])
    leaves = list(two_leaves)
    for j in range(K):
        if (root!=j) and (j not in level1) and (j not in two_leaves):
            leaves.append(j)
    C_init = []
    beta_hat = beta_hat[:, tuple(leaves)]
    for i in range(beta_hat.shape[0]):
        C_init.append(np.argmax(beta_hat[i]))
    C_init = np.array(C_init).astype(np.intc)
    return C_init

def initialize_trunc(key, corpus, tree, K, V, alpha, iters=600):
    _, J, K = tree.size
    beta_hat_trunc, _, _ = get_LDA_results(key, corpus, J, K, V, alpha=alpha, iterations=iters)
    labels1 = initialize1(beta_hat_trunc, tree)
    return labels1

def initialize_subset(key, corpus, tree, K, V, alpha, iters=600):
    _, J, K = tree.size
    _, beta_hat_subset, beta_hat = get_LDA_results(key, corpus, J, K, V, alpha=alpha, iterations=iters)
    labels2 = initialize2(beta_hat_subset, beta_hat)
    return labels2

def initialize_level(key, corpus, V, alpha=0.05, iters=600):
    # only for tree2 as our case
    # assumes corpus has doc_ids from 0 to (m-1)
    
    model = document_reader()
    model.process(corpus)
    
    tree1 = Tree({0: [1,2]})
    tree2 = Tree({0: [1], 1:[2,3]})
    C1 = initialize_trunc(key, corpus, tree1, tree1.size[2], V, alpha=alpha, iters=iters)

    ids_c0 = np.where(C1 == 0)[0]
    ids_c1 = np.where(C1 == 1)[0]

    corpus0 = model.get_docs_by_id(ids_c0, id_type='new')
    corpus1 = model.get_docs_by_id(ids_c1, id_type='new')

    model0 = document_reader()
    model0.process(corpus0)
    corpus0 = model0.corpus

    model1 = document_reader()
    model1.process(corpus1)
    corpus1 = model1.corpus

    
    C20 = initialize_trunc(key, corpus0, tree2, tree2.size[2], V, alpha=alpha, iters=iters)
    C21 = initialize_trunc(key, corpus1, tree2, tree2.size[2], V, alpha=alpha, iters=iters)

    c_init = np.zeros(model.n_docs, dtype=np.int32)
    c_init[ids_c0] = C20
    c_init[ids_c1] = C21 + 2 # account for level2
    c_init = jnp.array(c_init, dtype=jnp.int32)
    return c_init



# generalize the initialize2 to general trees
class Tree2:
    '''
    implements operations for tree
     - may / may not be balanced
     - input T is the dicitonary representation of the tree
     - 0 is root
    '''
    def __init__(self,T={0: []}):
        self.T = T
        nodes = set()
        for key in T:
            nodes.add(key)
            for val in T[key]:
                nodes.add(val)
        self.nodes = sorted(list(nodes))
        self.parents = self.parents_()
        self.K = len(self.nodes)
    
    def parents_(self):
        parents = {}
        for key in self.T:
            for val in self.T[key]:
                parents[val] = key
        return parents
    
    def add_node(self, edge):
        # edge = (parent, child)
        parent, child = edge
        if parent not in self.nodes:
            raise ValueError('Parent not in nodes')
        if child in self.nodes:
            raise ValueError('Child already in nodes')
        if parent in self.T.keys():
            self.T[parent] += [child]
        else:
            self.T[parent] = [child]
        self.nodes.append(child)
        self.nodes = sorted(self.nodes)
        self.parents = self.parents_()
        self.K = len(self.nodes)
            
    
    def children_(self, idx):
        return self.T[idx]
    
    def siblings_(self, idx):
        parent = self.parents[idx]
        return self.T[parent]
    
    def leaves_(self):
        # all nodes - nodes with children (set diff)
        nodes_with_children = self.T.keys()
        return list(set(self.nodes).difference(set(nodes_with_children)))
    
    def paths(self):
        '''
        return collection of paths as indices of nodes along them
        '''
        paths = []
        for leaf in self.leaves_():
            path = [leaf]
            while self.parents[leaf] in self.parents:
                path.append(self.parents[leaf])
                leaf = self.parents[leaf]
            path.append(0)
            path.reverse()
            paths.append(path)
        return paths
    
    def path_to_node(self, node):
        if node not in self.nodes:
            raise ValueError('node not in tree')
        path = []
        while node != 0:
            path.append(node)
            node = self.parents[node]
        path.append(0)
        path.reverse()
        other_nodes = set(self.nodes).difference(set(path))
        if len(other_nodes) > 0:
            other_nodes = list(other_nodes)
        else:
            other_nodes = []
        return path, other_nodes

def starts_with(path1, path2):
    return Counter(path1) == Counter(path2[0:len(path1)])

def is_tree_subset(tree_a, tree_b):
    '''
    check if tree_a is a sub-tree of tree_b
    '''
    tree_subset = False
    opt_perm = None
    
    nodes_a = tree_a.nodes
    K_a = len(nodes_a)
    K_b = len(tree_b.nodes)
    
    if K_a > K_b:
        return (False, None)
    
    for perm in permutations(np.arange(1, K_b)):
        new_nodes = {}
        tree_sub = True
        for i, node in enumerate(nodes_a):
            if node == 0:
                new_nodes[node] = 0
            else:
                new_nodes[node] = perm[i-1]
                
        # nodes_a[i] -> perm[i] i=1,...,Ka
        paths2 = tree_b.paths()
        
        for path1 in tree_a.paths():
            path1_new = [new_nodes[node] for node in path1]
            # is path1 a subset of at least one path of tree_b?
            check = [starts_with(path1_new, path2) for path2 in paths2]
            if np.any(check) == False:
                tree_sub = False
                break
            else:
                # remove path2 since it has been used once
                path2 = paths2[np.where(np.array(check) == True)[0][0]]
                paths2.remove(path2)
        if tree_sub is True:
            tree_subset = True
            opt_perm = new_nodes
            break
        
    return tree_subset, opt_perm

    
def initialize2_new(beta_hat_subset, beta_hat, tree):
    # tree is class Tree (not Tree2)
    K = beta_hat.shape[1]
    root = beta_hat_subset.sum(axis=0).argmax()
    new_tree = Tree2({0: []})
    dict_nodes = {0: root}
    
    for new_node_id in range(1, K):

        nodes = new_tree.nodes
        max_counts = -1
        counts_temp = 0
        
        for node in nodes:
            # check if adding a child to this node keeps it a sub-tree of tree
            new_tree_temp = deepcopy(new_tree)
            new_tree_temp.add_node((node, new_node_id))
            
            if is_tree_subset(new_tree_temp, tree)[0] == True:
                counts_temp += 1
                # if yes, then proceed - collect (x, counts)
                path, other_nodes = new_tree.path_to_node(node)
                print(path)
                containing = [dict_nodes[node] for node in path]
                leave_out = [dict_nodes[node] for node in other_nodes]
                #print(f'containing = {containing}')
                #print(f'leave out = {leave_out}')
                x, count = find_children2(beta_hat_subset, containing, leave_out)
                print(f'idx {x} with count {count}')
                
                if count > max_counts:
                    max_counts = count
                    chosen_node = node
                    chosen_x = x


        new_tree.add_node((chosen_node, new_node_id))
        dict_nodes[new_node_id] = chosen_x
        print(f'new node added: {new_tree.T}, counts = {counts_temp}')
    leaves = new_tree.leaves_()
    original_leaves = [dict_nodes[node] for node in leaves]
    C_init = []
    beta_hat = beta_hat[:, tuple(original_leaves)]
    for i in range(beta_hat.shape[0]):
        C_init.append(np.argmax(beta_hat[i]))
    C_init = np.array(C_init).astype(np.intc)
    return C_init

def find_children2(indices, parent_history, except_nodes):
    # except_nodes may be those from a different path
    # pick docs containing parent_history
    sub_indices = indices.copy()
    sub_indices = np.array(sub_indices)
    for node in parent_history:
        sub_indices = subset_docs(sub_indices, node, 1)
    sums = sub_indices.sum(axis=0)
    # pick the top n_chil without those in parent_history and except_nodes
    for node in parent_history:
        sums[node] = -1
    for node in except_nodes:
        sums[node] = -1
    return np.argmax(sums), np.max(sums)