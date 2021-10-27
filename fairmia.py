from copy import deepcopy
import networkx as ntwx
from networkx.algorithms.operators.unary import reverse
import numpy as np
import pandas as pd
import random
from itertools import combinations
import pickle
from im import icm_simulation, greedy_influence_max, group_spread
from mia import MIA, getMIIA, getalpha, getap, getMIOA, getMIIA, getallap
from config import *


def SubMIA(network, k, theta, attrib_name, attrib_val):
    """MIA algorithm for influence maximization with MiniMax fairness - Algorithm4"""
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
        g_nodes = []
        for node in list(network.nodes()):
            if network.nodes[node][attrib_name] == attrib_val :
                g_nodes.append(node)
        nodes_path = ntwx.dijkstra_path(network, v, g_nodes[0], weight='weight')
        g_graph = network.subgraph(g_nodes+nodes_path)
        MIIAv_g = getMIIA(g_graph, v, 0, digraph = True)
        #print(list(MIIAv_g.edges()))
        for u in list(MIIAv_g.nodes()):
            Incinf_dict[u] = Incinf_dict[u] + getalpha(v, u, S, MIIAv_g, 0) * (1 - getap(u, S, MIIAv_g))
        
    print("Initialization Completed")
    #print(attrib_name)
    #print(attrib_val)
    #print(Incinf_dict)
    for i in range(1, k+1):
        g_nodes = []
        for node in list(network.nodes()):
            if network.nodes[node][attrib_name] == attrib_val :
                g_nodes.append(node)
        
        u = max(Incinf_dict, key = Incinf_dict.get)
        MIOA = getMIOA(network, u, theta, digraph = True)
        for v in list(MIOA.nodes()):
            MIIAv = getMIIA(network, v, theta, digraph = True)
            ap_all = getallap(S, MIIAv)
            for w in list(MIIAv.nodes()):
                Incinf_dict[w] = Incinf_dict[w] - getalpha(v, w, S, MIIAv, theta)*(1 - ap_all[w])
                #if network.nodes[w][attrib] == group:
                #    Incinf_dict_group[w] = Incinf_dict_group[w] - getalpha(v, w, S, MIIAv, theta)*(1 - ap_all[w])
            nodes_path = ntwx.dijkstra_path(network, v, g_nodes[0], weight='weight')
            g_graph = network.subgraph(g_nodes+nodes_path)
            #g_nodes.append(v)
            #g_graph = network.subgraph(g_nodes)
            MIIAv_g = getMIIA(g_graph, v, 0, digraph = True)
            ap_all_g = getallap(S, MIIAv_g)
            for w in list(MIIAv_g.nodes()):
                Incinf_dict[w] = Incinf_dict[w] - getalpha(v, w, S, MIIAv_g, 0)*(1 - ap_all_g[w])

        S.append(u)
        #print(S)
        for v in list(MIOA.nodes()):
            MIIAv = getMIIA(network, v, theta, digraph = True)
            ap_all = getallap(S, MIIAv)
            for w in list(MIIAv.nodes()):
                Incinf_dict[w] = Incinf_dict[w] + getalpha(v, w, S, MIIAv, theta)*(1 - ap_all[w])
                #if network.nodes[w][attrib] == group:
                #    Incinf_dict_group[w] = Incinf_dict_group[w] - getalpha(v, w, S, MIIAv, theta)*(1 - ap_all[w])
            MIIAv_g = getMIIA(g_graph, v, 0, digraph = True)
            ap_all_g = getallap(S, MIIAv_g)
            for w in list(MIIAv_g.nodes()):
                Incinf_dict[w] = Incinf_dict[w] + getalpha(v, w, S, MIIAv_g, 0)*(1 - ap_all_g[w])
    return S


def fmia(network, k, theta, attrib_dict):
    s = []
    g_spread_dict = dict()
    g_s = MIA(network, k, theta)
    ################################g_s = [37, 12, 298, 17, 264, 44, 21, 317, 320, 271, 77, 40, 287, 481, 39]
    g_spread_dict = group_spread(network, rand_diff_prob15, g_s, simulations, attrib_dict, edge_pp_lb, edge_pp_ub, rand_seed = 42)
    group = min(g_spread_dict, key = g_spread_dict.get)
    attrib = [key for key, value in attrib_dict.items() if group in value][0]
    #print(g_spread_dict)
    #print(group)
    #print(attrib)
    #return
    final_s = SubMIA(network, k, theta, attrib, group)
    #print(final_s)
    return final_s


def fmia_v2(network, k, theta, attrib_dict, mia_seed):
    s = []
    g_spread_dict = dict()
    g_s = mia_seed
    ################################g_s = [37, 12, 298, 17, 264, 44, 21, 317, 320, 271, 77, 40, 287, 481, 39]
    g_spread_dict = group_spread(network, rand_diff_prob15, g_s, simulations, attrib_dict, edge_pp_lb, edge_pp_ub, rand_seed = 42)
    group = min(g_spread_dict, key = g_spread_dict.get)
    attrib = [key for key, value in attrib_dict.items() if group in value][0]
    #print(g_spread_dict)
    #print(group)
    #print(attrib)
    #return
    final_s = SubMIA(network, k, theta, attrib, group)
    #print(final_s)
    return final_s