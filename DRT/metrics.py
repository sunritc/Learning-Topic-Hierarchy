import numpy as np
from DRT.tree_structure import *
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
import ot
import cvxpy as cp

def expanded_topics(topics, paths):
    # gives a 3d topic array: [path, depth, word] (I, J, V)
    I, J = paths.shape
    V = topics.shape[1]
    topics_expanded = np.zeros((I, J, V))
    for i in range(I):
        topics_expanded[i] = topics[tuple(paths[i]),]
    return topics_expanded

def are_permutation(arr1, arr2):
    # checks if arr1 is permutation of arr2
    if set(arr1) == set(arr2):
        return 1
    else:
        return 0
    
def correct_structure(theta, theta_hat, tree):
    '''
    both theta and theta hat are (K,V)
    tree is the class in tree_structure
    '''
    root = 0
    # find the otpimal permutation
    distances = pairwise_distances(theta, theta_hat)
    _, perm = linear_sum_assignment(distances)
    leaves = tree.leaves
    # col_ind is the permutation, to theta[i] is assigned to theta_hat[perm[i]]
    def correct_children(node):
        # check if the set of children of node (in theta)
        # matches the set of children of corresponding node (in theta_hat)
        if node in leaves:
            return 1
        if perm[node] in leaves:
            return 0
        children_idx = tree.children_(node)
        children_idx2 = tree.children_(perm[node])
        perm_children_idx = [perm[i] for i in children_idx2]
        permutation_children = are_permutation(children_idx, perm_children_idx)

        incorrect_children = np.any([correct_children(i)==0 for i in children_idx])
        if incorrect_children or (permutation_children == 0):
            return 0
        else:
            return 1
    return correct_children(root)

def correct_structure_tree1(theta, theta_hat, tree):
    '''
    INCORRCT
    in tree1, only check for root
    '''
    root = 0
    # find the otpimal permutation
    distances = pairwise_distances(theta, theta_hat)
    _, perm = linear_sum_assignment(distances)
    if perm[0] == 0:
        if are_permutation([perm[1], perm[3]], [1,3]) or are_permutation([perm[1], perm[3]], [2,4]):
            return 1
        else:
            return 0
    else:
        return 0
    
def topic_metric(topics1, topics2, wt1=None, wt2=None, p=1):
    '''
    topics1 (K1, V)
    topics2 (K2, V)
    computes W_p distance between Sum_k wt1[k] delta_topics1[k] and Sum_k wt2[k] delta_topics2[k]
    p is either 1 or 2
    wt1/2 could be estimated alpha_bars if required later
    '''
    if topics1.shape[1] != topics2.shape[1]:
        raise ValueError("Vocab sizes not compatible")
    K1, K2 = topics1.shape[0], topics2.shape[0]
    if wt1 is None:
        wt1 = np.ones(K1) / K1
    else:
        wt1 = np.array(wt1)
        wt1 = wt1 / wt1.sum()
    if wt2 is None:
        wt2 = np.ones(K2) / K2
    else:
        wt2 = np.array(wt2)
        wt2 = wt2 / wt2.sum()
        
    pair_distances = np.power(pairwise_distances(topics1, topics2), p)
    W = ot.emd(wt1, wt2, pair_distances)
    optimal_cost = np.power(np.sum(W * pair_distances), 1/p)
    return optimal_cost

def tree_topic_metric(Topics1, pi1, Topics2, pi2, p=1):
    # Topics is (I, J, V) in expanded form
    # pi is length I
    I1, J1, V1 = Topics1.shape
    I2, J2, V2 = Topics2.shape
    
    if V1 != V2:
        raise ValueError("Incompatible V")
    if len(pi1) != I1:
        raise ValueError("pi1 incorrect size")
    if len(pi2) != I2:
        raise ValueError("pi2 incorrect size")
    distances = np.zeros((I1, I2))
    for i1 in range(I1):
        for i2 in range(I2):
            distances[i1,i2] = topic_metric(Topics1[i1], Topics2[i2], p=p)**p
    #print(distances)
    W = ot.emd(pi1, pi2, distances)
    optimal_cost = np.power(np.sum(W * distances), 1/p)
    return optimal_cost / V1
    
def topic_metric_L2(topics1, topics2):
    if topics1.shape[1] != topics2.shape[1]:
        raise ValueError("Incorrect V")
    if topics1.shape[0] != topics2.shape[0]:
        raise ValueError("Topics must have same K")
    distances = pairwise_distances(topics1, topics2)
    row_ind, col_ind = linear_sum_assignment(distances)
    return distances[row_ind, col_ind].sum() / topics1.shape[0]

def tree_topic_metric_L2(Topics1, Topics2):
    I, J, V = Topics1.shape
    I2, J2, V2 = Topics2.shape
    if I != I2:
        raise ValueError("Incorrect I")
    if J != J2:
        raise ValueError("Incorrect J")
    if V != V2:
        raise ValueError("Incorrect V")
    distances = np.zeros((I,I))
    for i1 in range(I):
        for i2 in range(I):
            distances[i1,i2] = topic_metric_L2(Topics1[i1], Topics2[i2])
    row_ind, col_ind = linear_sum_assignment(distances)
    return distances[row_ind, col_ind].sum() / V
    
def parallel_metrics(topics, theta_samples, pi, pi_samples, paths):
    Topics = expanded_topics(topics, paths)
    def f(i):
        Theta_hat = expanded_topics(theta_samples[i], paths)
        metricW = tree_topic_metric(Topics, pi, Theta_hat, pi_samples[i])
        metricL2 = tree_topic_metric_L2(Topics, Theta_hat)
        return [metricW, metricL2]
    distances = [f(i) for i in range(theta_samples.shape[0])]
    distances = np.array(distances).reshape((len(theta_samples), 2))
    return distances.mean(axis=0)
    
def parallel_metrics2(topics, theta_samples, pi, pi_samples, paths):
    topics = np.array(topics)
    theta_samples = np.array(theta_samples)
    pi = np.array(pi)
    pi_samples = np.array(pi_samples)
    paths = np.array(paths)
    Topics = expanded_topics(topics, paths)
    def f(i):
        Theta_hat = expanded_topics(theta_samples[i], paths)
        metricW = tree_topic_metric(Topics, pi, Theta_hat, pi_samples[i])
        metricL2 = tree_topic_metric_L2(Topics, Theta_hat)
        return [metricW, metricL2]
    distances = [f(i) for i in range(theta_samples.shape[0])]
    distances = np.array(distances).reshape((len(theta_samples), 2))
    return distances

def explore_simplex(topics):
    K, d = topics.shape
    distances = pairwise_distances(topics)
    distances = distances[np.triu_indices(K, k = 1)]
    max_side = np.max(distances)
    min_side = np.min(distances)
    projs = []
    for k in range(K):
        idx = [i for i in range(K) if i != k]
        x = topics[k]
        remaining = topics[tuple(idx),]
        projs.append(proj_simplex(x, remaining)[0])
    min_proj = np.min(np.array(projs))
    return {'dimension': K-1,
            'min side': min_side,
            'max side': max_side,
            'min width': min_proj}

def distance_simplex(topics1, topics2):
    distances = pairwise_distances(topics1, topics2)
    return max(np.max(np.min(distances, axis=0)), np.max(np.min(distances, axis=1)))

def proj_simplex(x, G):
    # returns distance, projected point
    K, d = G.shape
    if len(x) != d:
        raise ValueError('incorrect dim')

    beta = cp.Variable(K, name='beta')

    C = np.ones(K)

    constraints = [beta >= np.zeros(K),
                   C @ beta == 1.]

    prob = cp.Problem(cp.Minimize(cp.norm(x - G.T @ beta, 2)), constraints)
    prob.solve(solver=cp.ECOS)
    return prob.value, G.T @ beta.value


def proj_aff(x, G):
    K, d = G.shape
    if len(x) != d:
        raise ValueError('incorrect dim')
    beta = cp.Variable(K, name='beta')

    C = np.ones(K)

    constraints = [C @ beta == 1.]

    prob = cp.Problem(cp.Minimize(cp.norm(x - G.T @ beta, 2)), constraints)
    prob.solve(solver=cp.ECOS)
    return prob.value, G.T @ beta.value