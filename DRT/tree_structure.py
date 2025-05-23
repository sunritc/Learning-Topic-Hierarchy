'''
Created by SUNRIT on Jan 17, 2024
Implements directed rooted tree structure
use Tree class - implements levels, membership, parents, siblings, paths etc
For now: each path of tree must be same depth
However, it need not be balanced
Tree takes a dictionary as input (this is the adjacency list of rooted tree)
keep 0 as the root
'''

import numpy as np
import matplotlib.pyplot as plt

def tree_size(branching, depth):
    J = depth
    K = (branching**depth - 1) // (branching - 1)
    I = branching ** (depth - 1)
    return (I, J, K)

def get_T(branching, depth):
    '''
    get the tree T from branching, depth in complete balanced case
    '''
    T = {}
    I, J, K = tree_size(branching, depth)
    idx = [i for i in range(K)]
    for node in idx:
        if branching*node+1 < K:
            T[node] = [node*branching+1+j for j in range(branching)]
    return T

class Tree:
    '''
    implements operations for tree
     - may / may not be balanced
     - each path has same depth
     - input T is the representation of the tree
     - 0 is root and number nodes in ascending order
    '''
    def __init__(self,T):
        self.T = T
        nodes = set()
        for key in T:
            nodes.add(key)
            for val in T[key]:
                nodes.add(val)
        self.nodes = sorted(list(nodes))
        self.parents = self.parents_()
        self.levels = self.levels_()
        J = max(self.levels.keys()) + 1
        self.leaves = self.levels[J-1]
        K = len(self.nodes)
        I = len(self.leaves)
        self.size = (I, J, K)
        self.membership = self.membership_()
        
        
    def levels_(self):
        levels = {}
        left = set(self.nodes)
        levels[0] = [0]
        left.remove(0)
        layer = 1
        while len(left) > 0:
            last_level = levels[layer-1]
            level_nodes = set()
            for node_last in last_level:
                for node in self.T[node_last]:
                    level_nodes.add(node)
                    left.remove(node)
            levels[layer] = list(level_nodes)
            layer += 1
        return levels
        
        
    def membership_(self):
        levels = self.levels
        members = {}
        layer = self.size[1] - 1
        for node in levels[layer]:
            members[node] = 1
        
        while layer > 0:
            layer -= 1
            for node in levels[layer]:
                children = self.T[node]
                members[node] = sum([members[child] for child in children])
        return members
        
    
    def parents_(self):
        parents = {}
        for key in self.T:
            for val in self.T[key]:
                parents[val] = key
        return parents
    
    def children_(self, idx):
        return self.T[idx]
    
    def siblings_(self, idx):
        parent = self.parents[idx]
        return self.T[parent]
    
    def leaves_(self):
        # all nodes - nodes with children (set diff)
        nodes_with_children = self.T.keys()
        a = list(set(self.nodes).difference(set(nodes_with_children)))
        return a
    
    def paths(self):
        '''
        return collection of paths as indices of nodes along them
        '''
        paths = []
        for leaf in self.leaves:
            path = [leaf]
            while self.parents[leaf] in self.parents:
                path.append(self.parents[leaf])
                leaf = self.parents[leaf]
            path.append(0)
            path.reverse()
            paths.append(path)
        return paths
        
    def plot(self, ax):
        reference = {}
        layer = self.size[1]-1
        y_ = 0
        leaves = self.leaves
        leaves.sort()
        x_ = 0
        for leaf in leaves:
            reference[leaf] = (x_, y_)
            x_ += 1
        
        while layer > 0:
            layer -= 1
            y_ += 2
            nodes = self.levels[layer]
            for node in nodes:
                children = self.T[node]
                X_children = [reference[child][0] for child in children]
                reference[node] = (sum(X_children) / len(X_children), y_)
        
        node = 0
        children = self.T[node]
        X_children = [reference[child][0] for child in children]
        reference[node] = (sum(X_children) / len(X_children), y_)
        
        ax = ax or plt.gca()
        for node in self.T:
            for child in self.T[node]:
                line, = ax.plot([reference[node][0], reference[child][0]], [reference[node][1], reference[child][1]], color='black', alpha=0.3, linestyle='dotted')
                
        for node in reference:
            line = ax.scatter(reference[node][0], reference[node][1], color='red', alpha=0.2, s=60)
            ax.text(reference[node][0], reference[node][1], str(node), fontsize=15)
        ax.axis('off')
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth(1)  
        
        return line
        
def cpaths(tree):
    paths_ = tree.paths()
    I = len(paths_)
    J_ = [len(paths_[i]) for i in range(I)]
    J_max = np.max(J_)
    paths_return = np.ones((I, J_max), dtype=np.intc) * -1
    for i in range(I):
        for j in range(J_[i]):
            paths_return[i,j] = paths_[i][j]
    return paths_return