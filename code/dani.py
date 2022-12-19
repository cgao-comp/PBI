import time
import networkx as nx
import numpy as np
import math
from utils import *
from collections import defaultdict

def get_index(cascade):

    S = sort_cascade(cascade)
    CV = {}
    nodes = S[0]
    for i, node in enumerate(nodes):
        CV[node] = i+1

    return CV

def DANI(N, cascades, K):

    P = np.zeros([N, N])

    for cascade_dict in cascades:
        D = np.zeros([N, N])
        cascade = trans_list(cascade_dict)
        S = sort_cascade(cascade)
        CV = get_index(cascade)
        nodes = S[0]

        for i in range(len(nodes)):
            u = nodes[i]
            for j in range(i+1,len(nodes)):
                v = nodes[j]
                D[u][v] = 1 / (CV[v] * (CV[v] - CV[u]))

        for i in range(N):
            D[i] = D[i] / np.sum(D[i])
            for j in range(N):
                if D[i][j] > 0 :
                    P[(i,j)] += D[i][j]

    P_dict = defaultdict(float)
    for i in range(N):
        P[i] = P[i] / np.sum(P[i])
        for j in range(N):
            if i == j:
                continue
            else:
                if P[i][j] > 0:
                    P_dict[(i,j)] = P[i][j]

    node_status_set = {}
    for i in range(N):
        node_status_set[i] = set()
        for c_i,cascade_dict in enumerate(cascades):
            cascade = trans_list(cascade_dict)
            if i in cascade[0]:
                node_status_set[i].add(c_i)

    A = {}
    for u in range(N):
        for v in range(u+1,N):
            A[(u,v)] = len(node_status_set[u] & node_status_set[v]) / len(node_status_set[u] | node_status_set[v]) * (P_dict[(u,v)]+P_dict[(v,u)])

    result = []
    for key in A.keys():
        result.append((key,A[key]))

    def takeSecond(elem):
        return elem[1]

    result.sort(key=takeSecond,reverse=True)

    IG = nx.Graph()

    i=0
    while(len(IG.edges) < K):
        IG.add_edge(result[i][0][0],result[i][0][1])
        i+=1
        if i >= len(result):
            break
    return IG, result, A




