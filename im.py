from copy import deepcopy
import networkx as ntwx
import numpy as np
import pandas as pd
import random
from itertools import combinations
import pickle
from config import *


def icm_simulation(network, diffusion_prob, seed_nodes, mc_sumulations, edge_pp_lb, edge_pp_ub, rand_seed = 42):
    """Information Diffusion simulation using Independent Cascade Model using diffusion probability
       Returns average spread and the list of diffused/propagated nodes from last simulation"""
    spread_dict = dict()
    np.random.seed(rand_seed)
    for simulation in range(mc_sumulations):
        curr_active, cascade_list = deepcopy(seed_nodes), deepcopy(seed_nodes)
        iteration = 0
        while curr_active:
            n_list = [ntwx.function.neighbors(network, node) for node in curr_active]
            n_list_flat = [node for sublist in n_list for node in sublist]
            s_prob = np.random.uniform(0, 1, len(n_list_flat)) < diffusion_prob
            s_n_list = list(np.extract(s_prob, n_list_flat))
            curr_active = list(set(s_n_list) - set(cascade_list))
            cascade_list = cascade_list + curr_active
            iteration = iteration + 1
        spread_dict[iteration] = len(cascade_list)
    return np.mean(list(spread_dict.values())), cascade_list


def greedy_influence_max(network, num_seed, model, rand_diff_prob, b, theta, mc_sumulations):
    """Function to obtain seed nodes and also the spread using greedy algorithm for influence maximization
       Returns the best spread and also the seed nodes that result in best influence/spread."""
    all_nodes = deepcopy(list(set(network.nodes())))
    best_spread = 0
    best_seed = []
    while len(best_seed) < num_seed:
        #print(len(best_seed))
        node_spead_dict = dict()
        for node in all_nodes:
            if node not in best_seed:
                temp_list = deepcopy(best_seed)
                temp_list.append(node)
                if model == 'icm':
                    spread, _ = icm_simulation(network = network, diffusion_prob = rand_diff_prob, seed_nodes = temp_list, mc_sumulations = mc_sumulations, rand_seed = 42)
                elif model == 'ltm':
                    pass
                    #spread, _ = ltm_simulation_v1(network = network, b = b, theta =  theta, seed_nodes = temp_list, mc_sumulations = mc_sumulations, rand_seed = 42)
                node_spead_dict[node] = spread
        best_node = max(node_spead_dict, key=node_spead_dict.get)
        best_spread = node_spead_dict[best_node]
        if best_node not in best_seed:
            best_seed.append(best_node)
    return best_spread, best_seed


def group_spread(network, diffusion_prob, seed_nodes, mc_sumulations, attrib_dict, edge_pp_lb, edge_pp_ub, rand_seed = 42):
    """Information Diffusion simulation using Independent Cascade Model using diffusion probability
       Returns average group normalized spread (group_spread/number of nodes of each group) for each group"""
    spread_dict = dict()
    np.random.seed(rand_seed)
    for attrib_name in attrib_dict.keys():
        for attrib_val in attrib_dict[attrib_name]:
            spread_dict[attrib_val] = 0
    for simulation in range(mc_sumulations):
        curr_active, cascade_list = deepcopy(seed_nodes), deepcopy(seed_nodes)
        iteration = 0
        while curr_active:
            n_list = [ntwx.function.neighbors(network, node) for node in curr_active]
            n_list_flat = [node for sublist in n_list for node in sublist]
            s_prob = np.random.uniform(0, 1, len(n_list_flat)) < diffusion_prob
            s_n_list = list(np.extract(s_prob, n_list_flat))
            curr_active = list(set(s_n_list) - set(cascade_list))
            cascade_list = cascade_list + curr_active
            iteration = iteration + 1
        for attrib_name in attrib_dict.keys():
            for attrib_val in attrib_dict[attrib_name]:
                gcascade_list = []
                for item in cascade_list:
                    if network.nodes[item][attrib_name] == attrib_val:
                        gcascade_list.append(item)
                c_i = 0
                for item in list(network.nodes()):
                    if network.nodes[item][attrib_name] == attrib_val:
                        c_i = c_i + 1
                if c_i == 0:
                    c_i = 1 ## To avoid divide by zero in the below line of code
                #print(attrib_name)
                #print(attrib_val)
                #print(len(gcascade_list))
                #print(c_i)
                #print("\n")
                spread_dict[attrib_val] = spread_dict[attrib_val] + len(gcascade_list)/c_i
    final_spread_dict = {k: v / mc_sumulations for k, v in spread_dict.items()}
    #print(final_spread_dict)
    return final_spread_dict