import pickle
import networkx as nx
import numpy as np


def read_cascades(path):

    cascades = []
    with open(path,'r') as f:
        all_cascades = f.readlines()

        for cascade in all_cascades:

            cascade_dict = {}
            cascade = cascade.strip()
            cascade = cascade.split(',')

            max_T  = 0
            for i in range(0,len(cascade),2):
                cascade_dict[int(cascade[i])] = float(cascade[i+1])
                if float(cascade[i+1]) > max_T:
                    max_T = float(cascade[i+1])

            cascade_dict["T"] = max_T
            cascades.append(cascade_dict)

    return cascades

def sort_cascade(cascade):

    node_id = np.array(cascade[0])
    infected_time = np.array(cascade[1])
    index = np.argsort(infected_time)
    sorted_node_id = node_id[index].tolist()
    sorted_infected_time = infected_time[index].tolist()
    return [sorted_node_id, sorted_infected_time]

def sort_cascades(cascades):

    sort_Cascades = []
    for C in cascades:
        if len(C[0]) > 1:
            sort_C = sort_cascade(C)
            sort_Cascades.append(sort_C)

    return sort_Cascades

def load_pickle(path):
    with open(path,"rb") as f:
        return pickle.load(f)

def write_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def compute_F_score(IG,RG , is_directed = False):

    if not is_directed:
        IG = nx.Graph(IG)
        RG = nx.Graph(RG)

    ig_edges = IG.edges
    rg_edges = RG.edges

    TP = 0.0
    FP = 0.0
    FN = 0.0

    for (i,j) in ig_edges:
        if (i,j) in rg_edges or (j,i) in rg_edges:
            TP += 1.0
        else:
            FP += 1.0

    for (i,j) in rg_edges:
        if (i,j) not in ig_edges and (j,i) not in ig_edges:
            FN += 1.0

    P = TP / (TP+FP)
    R = TP / (TP+FN)

    return P,R,2*P*R / (P+R)


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
