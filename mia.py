from copy import deepcopy
import networkx as ntwx
import numpy as np
import pandas as pd
import random
from itertools import combinations
import matplotlib.pyplot as plt


def getpp(network, w, u):
    """Returns path propagation probability"""
    pp = 1
    path = ntwx.dijkstra_path(network, w, u, weight = 'weight')
    edges = list(ntwx.utils.misc.pairwise(path))
    for edge in edges:
        #print(network[edge[0]][edge[1]]['weight'])
        pp = pp*network[edge[0]][edge[1]]['weight']
    return pp


def getMIIA(graph, v, theta, digraph = True):
    "Returns MAXIMUM INFLUENCE IN-ARBORESCENCE"
    #nodes = list(dgraph.nodes())
    if digraph == True:
      miia = ntwx.DiGraph()
    else:
      miia = ntwx.Graph()
    mir_pairs = dict(ntwx.all_pairs_dijkstra(graph, weight='weight'))
    for val in mir_pairs.values():
        #print(val[1].values())
        if v in val[1].keys():
            #print(v)
            pp = getpp(graph, val[1][v][0], val[1][v][-1])
            if pp > theta:
                edges = list(ntwx.utils.misc.pairwise(val[1][v]))
                miia.add_edges_from(edges)
                #miia.add_nodes_from(val[1][v])
                #ntwx.draw(miia)
    for edge in miia.edges():
      miia[edge[0]][edge[1]]['weight'] = graph[edge[0]][edge[1]]['weight']
    return miia

def getMIOA(graph, v, theta, digraph = True):
    "Returns MAXIMUM INFLUENCE OUT-ARBORESCENCE"
    if digraph == True:
      mioa = ntwx.DiGraph()
    else:
      mioa = ntwx.Graph()
    mir_pairs = dict(ntwx.all_pairs_dijkstra(graph, weight='weight'))
    if v in mir_pairs.keys():
        for val in mir_pairs[v][1].values():
            pp = getpp(graph, val[0], val[-1])
            if pp > theta:
                #print(val)
                edges = list(ntwx.utils.misc.pairwise(val))
                #print(edges)
                mioa.add_edges_from(edges)
                #ntwx.draw(mioa)
    for edge in mioa.edges():
      mioa[edge[0]][edge[1]]['weight'] = graph[edge[0]][edge[1]]['weight']
    return mioa


def getap(u, S, MIIA):
    """Returns activation probability given a note and MIIA"""
    if len(S) == 0:
        return 0
    if u in S :
        return 1
    elif len(list(MIIA.predecessors(u))) == 0:
        return 0
    else:
        ap = 1
        #print(list(MIIA.predecessors(u)))
        for w in list(MIIA.predecessors(u)):
            #print("Node:", w)
            ap = 1 - ap*(1 - getap(w, S, MIIA)*getpp(MIIA, w, u))
            #print(ap)
            #print("\n")
    return ap

def getallap(S, MIIA):
    """Returns a dictionary activation probabilities for all nodes in MIIA"""
    ap_dict = dict()
    node_nb_dict = dict()
    for n in MIIA.nodes():
      node_nb_dict[n] = len(list(MIIA.predecessors(n)))
    #temp = list(ntwx.topological_sort(MIIA))
    #print(temp)
    #print(node_nb_dict)
    #print(sorted(node_nb_dict, key=node_nb_dict.get))
    for u in list(ntwx.topological_sort(MIIA)):
        if u in S :
            ap_dict[u] = 1
        elif node_nb_dict[u] == 0:
            ap_dict[u] = 0
        else:
            ap = 1
            #print(list(MIIA.predecessors(u)))
            #print(u)
            #print(list(MIIA.predecessors(u)))
            for w in list(MIIA.predecessors(u)):
                #print("\n")
                #print("Node:", w)
                #print(node_nb_dict[w])
                ap = ap*(1 - ap_dict[w]*getpp(MIIA, w, u))
                #print(ap)
                #print("\n")
            ap_dict[u] = 1 - ap
    return ap_dict


def getalpha(v, u, S, network, theta):
    """ Returns the value of aplha as per Algorithm3"""
    MIIA = getMIIA(network, v, theta, True)
    if v == u :
        return 1
    else:
        w = list(MIIA.successors(u))[0]
        if w in S :
            return 0
        else:
            alpha = getalpha(v, w, S, MIIA, theta)*getpp(MIIA, u, w)
            for u_dash in list(MIIA.predecessors(w)):
                alpha = alpha * (1 - getap(u_dash, S, MIIA)*getpp(MIIA, u_dash, w))
    return alpha



def MIA(network, k, theta):
    """MIA algorithm for influence maximization - Algorithm4"""
    S = []
    Incinf_dict = dict()
    ap_dict = dict()
    for v in list(network.nodes()):
        Incinf_dict[v] = 0
    for v in list(network.nodes()):
        MIIAv = getMIIA(network, v, theta, digraph = True)
        MIIA = MIIAv
        #MIOA = getMIOA(network, v, theta, digraph = True)
        #print(len(list(MIIA.nodes())))
        for u in list(MIIA.nodes()):
            Incinf_dict[u] = Incinf_dict[u] + getalpha(v, u, S, MIIAv, theta) * (1 - getap(u, S, MIIA))
            #print(Incinf_dict[u])

    print("Initialization Completed")
    for i in range(1, k+1):
        u = max(Incinf_dict, key = Incinf_dict.get)
        MIOA = getMIOA(network, u, theta, digraph = True)
        for v in list(MIOA.nodes()):
            MIIAv = getMIIA(network, v, theta, digraph = True)
            ap_all = getallap(S, MIIAv)
            for w in list(MIIAv.nodes()):
                Incinf_dict[w] = Incinf_dict[w] - getalpha(v, w, S, MIIAv, theta)*(1 - ap_all[w])
        S.append(u)
        #print(S)
        for v in list(MIOA.nodes()):
            MIIAv = getMIIA(network, v, theta, digraph = True)
            ap_all = getallap(S, MIIAv)
            for w in list(MIIAv.nodes()):
                Incinf_dict[w] = Incinf_dict[w] + getalpha(v, w, S, MIIAv, theta)*(1 - ap_all[w])
    return S





