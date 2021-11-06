from copy import deepcopy
import networkx as ntwx
import numpy as np
import pandas as pd
import random
from itertools import combinations
import pickle
from im import icm_simulation, greedy_influence_max, group_spread
from mia import MIA
import matplotlib.pyplot as plt
from config import *


def PoF(network, attrib_dict, S, Sf, diff_prob, simulations):
    """Returns price of fairness, given optimal seeds(vanilla IM) and fair seeds(Fair IM)
    for all attributes as a dictionary"""
    PoF_dict = dict()
    g_spread_dict = group_spread(network, diff_prob, S, simulations, attrib_dict, edge_pp_lb, edge_pp_ub, rand_seed = 42)
    g_spread_dict_fair = group_spread(network, diff_prob, Sf, simulations, attrib_dict, edge_pp_lb, edge_pp_ub, rand_seed = 42)
    for attrib_name in attrib_dict.keys():
        ginf_opt = []
        ginf_fair = []
        pofg_list = []
        for attrib_val in attrib_dict[attrib_name]:
            ginf_opt.append(g_spread_dict[attrib_val])
            ginf_fair.append(g_spread_dict_fair[attrib_val])
            if g_spread_dict_fair[attrib_val] == 0:
                pof_g = g_spread_dict[attrib_val]/(g_spread_dict_fair[attrib_val] + 0.01)
            else:
                pof_g = g_spread_dict[attrib_val]/g_spread_dict_fair[attrib_val]
            pofg_list.append(pof_g)
        #print(ginf_opt)
        #print(ginf_fair)
        PoF_dict[attrib_name] = np.average(pofg_list)
    return PoF_dict

def MInf(network, attrib_dict, Sf, diff_prob, simulations):
    """Returns minimum influence dictionary for each attribute"""
    MInf_dict = dict()
    g_spread_dict = group_spread(network, diff_prob, Sf, simulations, attrib_dict, edge_pp_lb, edge_pp_ub, rand_seed = 42)
    for attrib_name in attrib_dict.keys():
        ginf = []
        for attrib_val in attrib_dict[attrib_name]:
            ginf.append(g_spread_dict[attrib_val])
        MInf_dict[attrib_name] = np.min(ginf)
    return MInf_dict


def plot_pof(pof, title):
    """ Plots bar chart for Price of group fairness metric """
    plt.bar(range(len(pof)), pof.values(), align='center')
    plt.xticks(range(len(pof)), list(pof.keys()))
    plt.ylabel("Price of group fairness")
    plt.title(title)
    figname = title+str(pof)+str('.pdf')
    plt.savefig(figname, format = 'pdf')
    plt.show()
    plt.close()

def plot_minf(minf, minf_fair, title): # Reference : Stackoverflow https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
    """ Plots side-by-side bar chart for Min fraction influenced metric """
    min_c = dict()
    for item in minf_fair.keys():
        val_list = []
        val_list.append(minf[item])
        val_list.append(minf_fair[item])
        min_c[item] = val_list
    
    keys = [key for key in min_c.keys()]
    values = [value for value in min_c.values()]
    fig, ax = plt.subplots()
    
    ax.bar(np.arange(len(keys)), [value[0] for value in values],
        width=0.2, color='b', align='center')
    ax.bar(np.arange(len(keys)) + 0.2,
        [value[1] if len(value) == 2 else 0 for value in values],
        width=0.2, color='g', align='center')
    ax.set_xticklabels(keys)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_ylabel("Min fraction influenced")
    ax.set_title(title)
    colors = {'MIA':'blue', 'FairMIA':'green'}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    ax.legend(handles, labels)
    figname = title+str(minf)+str('.pdf')
    plt.savefig(figname, format = 'pdf')
    plt.show()
    plt.close()


def fairness_score(n,n_influenced):
    '''

    :param n: list   number of nodes in each group
    :param n_influenced: list   number of nodes influenced in each group

    :return: fairness score

    :EXAMPLE:
    fairness_score([40,30,20],[20,20,10])
    n can be a list of ones and n_influenced can be normalized influence for all groups.

    '''

    n_groups=len(n)
    if n_groups!=len(n_influenced):
        raise Exception('length of n,n_influenced should be same')
    percentage_of_influence=list(map(lambda x,y : round((y/x)*100,4) , n,n_influenced))
    mean= round(sum(percentage_of_influence)/n_groups,4)
    deviation=list(map(lambda x: abs(mean-x),percentage_of_influence))
    sum_of_deviation=round(sum(deviation),4)
    deviation=sum_of_deviation/mean
    return round(1-deviation/n_groups,4)

    

def FrScores(network, attrib_dict, Sf, diff_prob, simulations):
    """Returns fairness scores dictionary for each attribute by considering all groups of each attribute"""
    FrS_dict = dict()
    g_spread_dict = group_spread(network, diff_prob, Sf, simulations, attrib_dict, edge_pp_lb, edge_pp_ub, rand_seed = 42)
    for attrib_name in attrib_dict.keys():
        ginf = []
        for attrib_val in attrib_dict[attrib_name]:
            ginf.append(g_spread_dict[attrib_val])
        FrS_dict[attrib_name] = fairness_score([1]*len(ginf), ginf)
    return FrS_dict
