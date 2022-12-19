import time
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import math
import random as rand
from utils import *
from joblib import Parallel, delayed
from dani import *

def trans_list(cascade_dict):

    cascade = []
    nodes = []
    times = []

    for node in cascade_dict.keys():
        if node != "T":
            nodes.append(int(node))
            times.append(cascade_dict[node])

    cascade.append(nodes)
    cascade.append(times)

    return cascade

def get_infected_time(cascade, node):
    nodes = cascade[0]

    for i in range(len(nodes)):
        if nodes[i] == node:
            return

def hazard_function(alpha):

    return alpha

def log_survival_function(alpha, t_big, t_small):

    return -alpha*(t_big - t_small)

def init_G_with_cands(N,alpha, cascades, all_cands):
    G = nx.Graph()

    for (i,j) in all_cands:
        G.add_edge(i,j)

    for i in range(N):
        G.add_node(i)

    isolates_nodes = list(nx.isolates(G))

    for isolates_node in isolates_nodes:
        for cascade_dict in cascades:
            if isolates_node in cascade_dict.keys():
                for other_node in cascade_dict.keys():
                    if cascade_dict[other_node] < cascade_dict[isolates_node] and other_node != 'T':
                        G.add_edge(other_node,isolates_node)

    for cascade_dict in cascades:
        for node in cascade_dict.keys():
            if node != "T" and cascade_dict[node] > 0:
                node_neighbors = list(G.neighbors(node))
                if len(set(node_neighbors) & set(cascade_dict.keys())) == 0:
                    for other_node in cascade_dict.keys():
                        if cascade_dict[other_node] < cascade_dict[node]:
                            G.add_edge(other_node,node)

    cascades_H_dict = []

    cascades_weight_sum = []
    for cascade_dict in cascades:
        H_dict = defaultdict(int)
        weight_dict = {}
        for node in cascade_dict.keys():
            if node != "T" and cascade_dict[node] > 0:
                sum = 0
                H_dict[node] = 0
                node_neighbors = list(G.neighbors(node))

                for node_nei in node_neighbors:
                    if node_nei in cascade_dict.keys() and cascade_dict[node_nei] < cascade_dict[node]:
                        H_dict[node] += hazard_function(alpha)

                for other_node in cascade_dict.keys():
                    if other_node != 'T' and cascade_dict[other_node] < cascade_dict[node] and other_node in node_neighbors: #归一化用当前初始化图的归一化
                        sum += math.exp(-(cascade_dict[node] - cascade_dict[other_node]))

                weight_dict[node] = sum

        cascades_H_dict.append(H_dict)
        cascades_weight_sum.append(weight_dict)

    return G, cascades_H_dict, cascades_weight_sum

def mcmc(n, cascades, alpha, p, iter, error = -2 , seed = 0, all_cands = None):

    G, cascades_H_dict, cascades_weight_sum = init_G_with_cands(n, alpha, cascades, all_cands)
    graph_ratio_rem = math.log(1 - p) - math.log(p)
    graph_ratio_add = math.log(p) - math.log(1 - p)

    if seed !=0:
        rand.seed(seed)

    init_edges= set(G.edges())
    cand_edges = set(all_cands)
    total_edges_list = list(init_edges | cand_edges)
    tot_edges = len(total_edges_list)
    edge_list = np.zeros([tot_edges, 2], dtype=int)
    k = 0
    for (i, j) in init_edges:
        if i > j:
            edge_list[k] = [j, i]
        else:
            edge_list[k] = [i, j]
        k += 1

    number_of_edges = k

    for (i, j) in cand_edges:
        if (i, j) not in G.edges and (j, i) not in G.edges:
            edge_list[k] = [i, j]
            k += 1

    normal_weights = [{} for _ in range(len(cascades))]

    for it in range(iter):

        sample = rand.random()
        if sample < 0.5 and number_of_edges > 1:
            rand_ind = int(math.floor(rand.random() * number_of_edges))
            [i,j] = edge_list[rand_ind]
            had_edge = 1
            Q_l = math.log(tot_edges-number_of_edges + 1) - math.log(number_of_edges)
            graph_ratio_l = graph_ratio_rem
        elif sample > 0.5 and number_of_edges < tot_edges:
            rand_ind = number_of_edges + int(math.floor(rand.random() * (tot_edges-number_of_edges)))
            [i,j] = edge_list[rand_ind]
            had_edge = 0
            graph_ratio_l = graph_ratio_add
            Q_l = math.log(number_of_edges+1) - math.log(tot_edges-number_of_edges)
        else:
            continue

        log_S_sum = 0
        new_H_i_list  = [0 for _ in range(len(cascades))]
        new_H_j_list = [0 for _ in range(len(cascades))]
        log_H_sum = 0
        log_weight_prod = 0

        for c_i,cascade_dict in enumerate(cascades):

            if i in cascade_dict.keys() and j in cascade_dict.keys():
                if cascade_dict[i] < cascade_dict[j]:
                    new_H_j = hazard_function(alpha)
                    new_H_j_list[c_i] = new_H_j
                    log_S_sum += log_survival_function(alpha=alpha, t_big=cascade_dict[j], t_small=cascade_dict[i])

                    sum = cascades_weight_sum[c_i][j]

                    if (i,j) in normal_weights[c_i].keys():
                        log_weight_prod += math.log(normal_weights[c_i][(i,j)])
                    else:
                        p = math.exp(-(cascade_dict[j] - cascade_dict[i])) / sum
                        w = math.ceil(p * 10) / 10.0
                        normal_weights[c_i][(i, j)] = w
                        normal_weights[c_i][(j, i)] = w
                        log_weight_prod += math.log(w)

                    if had_edge == 1:
                        if cascades_H_dict[c_i][j] - new_H_j == 0:
                            log_H_sum += error - math.log(cascades_H_dict[c_i][j])
                        else:
                            log_H_sum += math.log(cascades_H_dict[c_i][j] - new_H_j) - math.log(cascades_H_dict[c_i][j])
                    else:
                        if cascades_H_dict[c_i][j] == 0 :
                            log_H_sum += math.log(cascades_H_dict[c_i][j] + new_H_j) - error
                        else:
                            log_H_sum += math.log(cascades_H_dict[c_i][j] + new_H_j) - math.log(cascades_H_dict[c_i][j])


                elif cascade_dict[i] > cascade_dict[j]:
                    new_H_i = hazard_function(alpha)
                    new_H_i_list[c_i] = new_H_i
                    log_S_sum += log_survival_function(alpha=alpha, t_big=cascade_dict[i], t_small=cascade_dict[j])

                    sum = cascades_weight_sum[c_i][i]

                    if (i,j) in normal_weights[c_i].keys():
                        log_weight_prod += math.log(normal_weights[c_i][(i,j)])
                    else:
                        p = math.exp(-(cascade_dict[i] - cascade_dict[j])) / sum
                        w = math.ceil(p*10) / 10.0
                        normal_weights[c_i][(i, j)] = w
                        normal_weights[c_i][(j, i)] = w
                        log_weight_prod += math.log(w)

                    if had_edge == 1:
                        if cascades_H_dict[c_i][i] - new_H_i == 0:
                            log_H_sum += error - math.log(cascades_H_dict[c_i][i])
                        else:
                            log_H_sum += math.log(cascades_H_dict[c_i][i]- new_H_i) - math.log(cascades_H_dict[c_i][i])
                    else:
                        if cascades_H_dict[c_i][i] == 0:
                            log_H_sum += math.log(cascades_H_dict[c_i][i] + new_H_i) - error
                        else:
                            log_H_sum += math.log(cascades_H_dict[c_i][i] + new_H_i) - math.log(cascades_H_dict[c_i][i])

            elif i in cascade_dict.keys() and j not in cascade_dict.keys():
                log_S_sum += log_survival_function(alpha=alpha, t_big=cascade_dict["T"], t_small=cascade_dict[i])

            elif j in cascade_dict.keys() and i not in cascade_dict.keys():
                log_S_sum += log_survival_function(alpha=alpha, t_big=cascade_dict["T"], t_small=cascade_dict[j])

        if had_edge == 1:
            log_ratio = -log_S_sum + graph_ratio_l + Q_l + log_H_sum - log_weight_prod
        else:
            log_ratio = log_S_sum + graph_ratio_l + Q_l + log_H_sum + log_weight_prod

        if math.log(rand.random()) < log_ratio:

            if had_edge == 1:
                for c_i in range(len(cascades)):

                    cascades_H_dict[c_i][i] -= new_H_i_list[c_i]
                    cascades_H_dict[c_i][j] -= new_H_j_list[c_i]

                edge_list[rand_ind] = edge_list[number_of_edges - 1]
                edge_list[number_of_edges - 1] = [i, j]
                number_of_edges -= 1
                if (i, j) in G.edges:
                    G.remove_edge(i, j)
                if (j, i) in G.edges:
                    G.remove_edge(j, i)
            else:
                for c_i in range(len(cascades)):
                    cascades_H_dict[c_i][i] += new_H_i_list[c_i]
                    cascades_H_dict[c_i][j] += new_H_j_list[c_i]

                edge_list[rand_ind] = edge_list[number_of_edges]
                edge_list[number_of_edges] = [i, j]
                number_of_edges += 1
                G.add_edge(i, j)

    return G

def paral_MCMC(N, cascades, alpha, p, iter, error = 1e-100, sample_size = 10, cpu_num = 10, all_cands = None):

    samples = Parallel(n_jobs=cpu_num)(
        delayed(mcmc)(N, cascades, alpha = alpha, p = p, iter = iter, error = error, seed = rep, all_cands = all_cands) for rep in range(sample_size))

    return samples

