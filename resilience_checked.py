import networkx as nx
import itertools
from typing import Iterable
import random
import pandas as pd
import numpy as np
import os
import scipy.special
import joblib
from randomization import randomize_hyper_df
from utilities import *
import sys,gc
import psutil
from psutil._common import bytes2human
from collections import Counter
def find_triangles(G):
    return set([tuple(sorted(set(triple))) for triple in set(frozenset((n,nbr,nbr2)) for n in G for nbr, nbr2 in itertools.combinations(G[n],2) if nbr in G[nbr2])])



def run_resilience_analysis_fast(size_lim=50, revenue_limit='overall', type_data='board', randomization=False, seed1=0, inverse_order=False,unweighted = False,remove_hubs1 = False):
    """
    Run resilience analysis on the given dataset.

    Parameters:
    - size_lim (int): The size limit for the dataset.
    - revenue_limit (str): The revenue limit for the dataset.
    - type_data (str): The type of data to analyze ('board' or other):'board', 'collaboration_diff_iso', or 'collaboration_max_iso'.
    - randomization (bool): Whether to perform randomization on the dataset.
    - seed1 (int): The seed value for randomization.
    - inverse_order (bool): if False, include all hyperlinks with order less than or equal to the given order. If True, include all hyperlinks with order greater than or equal to the given order.

    Returns:
    - dict_component (dict): A dictionary containing the number of nodes and available nodes for each order.
    - iso_connected (dict): A dictionary containing the ISO codes of connected components for each order.
    - dict_component_nodes (dict): A dictionary containing the node sets for each order.
    """
    
    if type_data == 'board':
        type_data_str = 'inconsistent_ISO'
    else:
        type_data_str = type_data

    df = get_df_hyper(size_lim, revenue_limit, type_data=type_data,unweighted=unweighted)
    if remove_hubs1:
        df = remove_hubs(df,50)
        type_data+='_no_hubs'
    df = df[df['order']>1]
    rand_str = ''
    if type(randomization)!=bool:
        df = randomize_hyper_df(df, seed1=seed1, method=randomization, interlocking_directors=True)
        rand_str = '_rand_'+str(randomization)+'_seed' + str(seed1)
    

    if unweighted:
        type_data = type_data+ '_unweighted'
        
    total_hyperlinks = df
    node_iso = total_hyperlinks.explode(['ISO','BvD']).drop_duplicates('BvD').set_index('BvD')['ISO'].to_dict()

    total_hyperlinks.drop_duplicates('BvD',inplace=True)
    total_hyperlinks.loc[:, 'links'] = total_hyperlinks['BvD'].apply(lambda x: list(itertools.combinations(x, 2)))
    order_links = total_hyperlinks.explode('links').drop_duplicates(['links','order'])
    dict_component = {}
    iso_connected = {}
    dict_component_nodes = {}
    order_list = range(2, size_lim+1)
    inverse_str = ''
    iso_set = set()
    previous_lcc = set()
    if inverse_order: 
        order_list = range(2, size_lim+1)[::-1]
        inverse_str = '_inverse'
    for k,order in enumerate(order_list):
        print(order)
        largest_cc, available_nodes = compute_largest_connected_component_fast_checked(order_links, order_limit = order, inverse_order=inverse_order,previous_lcc1 = previous_lcc, index = k )
        previous_lcc = largest_cc.copy()
        if largest_cc == set():
            iso_connected[order] = Counter()
            iso_set = 0
        else:
            
            iso_connected[order] = iso_connected[order] = Counter([node_iso[node] for node in list(largest_cc)])
        dict_component[order] = (len(largest_cc), available_nodes)
        dict_component_nodes[order] = largest_cc
        joblib.dump(dict_component, '/data/ceriaac/CORPNET_project/Data/' + type_data_str + '/resilience_checked/resilience_nodes' + str(size_lim) + '_' + revenue_limit + '_' + inverse_str + rand_str + type_data + '_fast.joblib')
        joblib.dump(iso_connected, '/data/ceriaac/CORPNET_project/Data/' + type_data_str + '/resilience_checked/resilience_iso' + str(size_lim) + '_' + revenue_limit + '_' + inverse_str + rand_str + type_data + '_fast.joblib')
        print('resiliance analysis has been done for order ' + str(order))
        
    #joblib.dump(dict_component_nodes, '/data/ceriaac/CORPNET_project/Data/' + type_data_str + '/resilience/resilience_node_set' + str(size_lim) + '_' + revenue_limit + '_' + inverse_str + rand_str + type_data + '.joblib')
    del total_hyperlinks,largest_cc
    return dict_component, iso_connected

def compute_largest_connected_component_fast_checked(order_links, order_limit, inverse_order = False, previous_lcc1=None,index = 0):
    """
    Computes the largest connected component in a graph of hyperlinks.

    Args:
        hyperlinks (pandas.DataFrame): DataFrame containing the hyperlinks data.
        max_order (int): Maximum order of hyperlinks to consider.
        largest_cc (set, optional): Set representing the largest connected component. Defaults to None.
        inverse_order (bool): if False, include all hyperlinks with order less than or equal to the given order. If True, include all hyperlinks with order greater than or equal to the given order.

    Returns:
        tuple: A tuple containing the set representing the largest connected component and the filtered hyperlinks DataFrame.
    """
    if inverse_order:
        available_links = order_links[order_links['order']<=order_limit]['links'].unique()
    else:
        available_links = order_links[order_links['order']>=order_limit]['links'].unique()
    if len(available_links) == 0:
        largest_cc = []
        available_nodes = 0
        G = nx.Graph()
    else:
        G = nx.Graph()
        G.add_edges_from(available_links)
        available_nodes = len(G.nodes)
        if index == 0:
            largest_cc = max(nx.connected_components(G), key=len)
        # largest_cc = max(nx.connected_components(G), key=len)
        else:
            if max([len(previous_lcc1.intersection(set(x))) for x in nx.connected_components(G)]) == 0:
                largest_cc = set()
            else: 
                largest_cc = max(nx.connected_components(G), key=lambda s: len(previous_lcc1.intersection(set(s))))
        
    return set(largest_cc), available_nodes
