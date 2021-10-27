from copy import deepcopy
import networkx as ntwx
import numpy as np
import pandas as pd
import random
from itertools import combinations
import pickle
from im import icm_simulation, greedy_influence_max
from mia import MIA
from fairmia import fmia
from utils import PoF, MInf, plot_pof, plot_minf
import NetworkGenerator as gen
from config import *


def main():        
    """main function that contains the entire pipeline or encapsulation of all other functions"""

    ### Using Simulated Antelope Valley network (Ref : https://github.com/bwilder0/fair_influmax_code_release/tree/master/networks )
    fgraph = pickle.load(open(graph_path, 'rb'))
    #rand_diff_prob15 = np.random.uniform(edge_pp_lb, edge_pp_ub)
    edge_probs = np.random.uniform(edge_pp_lb, edge_pp_ub, fgraph.number_of_edges()) ### Generate random edge propagation probabilities between 0 and 0.1
    for idx, edge in enumerate(fgraph.edges()):
        fgraph[edge[0]][edge[1]]['weight'] = edge_probs[idx]

    
    print("Results on Simulated Antelope Valley network :")
    print("Attributes of the graph are :", attrib_dict)
    
    final_s = MIA(fgraph, k, theta)
    print("Seeds for IM with MIA :", final_s)
    final_sf = fmia(fgraph, k, theta, attrib_dict)
    print("Seeds for IM with fair MIA :", final_sf)
    
    
    pof = PoF(fgraph, attrib_dict, final_s, final_sf, rand_diff_prob15, simulations)
    print("Price of group fairness :",pof)
    plot_pof(pof, "Simulated Antelope Valley network")
    
    minf_fair = MInf(fgraph, attrib_dict, final_sf, rand_diff_prob15, simulations)
    print("Min fraction influenced with fair MIA :",minf_fair)
    minf = MInf(fgraph, attrib_dict, final_s, rand_diff_prob15, simulations)
    print("Min fraction influence with MIA :",minf)
    plot_minf(minf, minf_fair, "Simulated Antelope Valley network")

  
    ### Using Synthetically generated Network
    hi = gen.barabasiAlbertAttributeModel(nodes, 4)
    hi.add_choice_attribute('Gender', ['male', 'female'], weights=[50, 50],d_index=3)
    hi.add_choice_attribute('Region', ['India', 'US'], weights=[90,10], d_index=2)
    hi.add_choice_attribute('Age', ['20-25', '50-59'], weights=[60, 40], d_index=1)

    G = hi.generate_graph()

    edge_probs_G = np.random.uniform(edge_pp_lb, edge_pp_ub, G.number_of_edges()) ### Generate random edge propagation probabilities between 0 and 0.1
    for idx, edge in enumerate(G.edges()):
        G[edge[0]][edge[1]]['weight'] = edge_probs_G[idx]

    print("Results on Synthetic network G :")
    print("Attributes of the graph G are :", attrib_dict_g)
    final_g = MIA(G, k, theta)
    print("Seeds for IM with MIA :", final_g)
    final_gf = fmia(G, k, theta, attrib_dict_g)
    print("Seeds for IM with fair MIA :", final_gf)
    
    pof_g = PoF(G, attrib_dict_g, final_g, final_gf, rand_diff_prob15, simulations)
    print("Price of group fairness :",pof_g)
    plot_pof(pof_g, "Synthetic Network")
    
    minf_fair_g = MInf(G, attrib_dict_g, final_gf, rand_diff_prob15, simulations)
    print("Min fraction influenced with fair MIA :",minf_fair_g)
    minf_g = MInf(G, attrib_dict_g, final_g, rand_diff_prob15, simulations)
    print("Min fraction influence with MIA :",minf_g)
    plot_minf(minf_g, minf_fair_g, "Synthetic Network")

if __name__ == '__main__':
    main()


