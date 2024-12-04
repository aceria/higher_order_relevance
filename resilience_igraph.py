import networkx as nx
import itertools
from typing import Iterable
from collections import Counter
import random
import pandas as pd
import numpy as np

import joblib
from randomization import randomize_hyper_df
from utilities import *
import os, psutil; 
import sys,gc
import psutil
from psutil._common import bytes2human
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def find_triangles(G):
    return set([tuple(sorted(set(triple))) for triple in set(frozenset((n,nbr,nbr2)) for n in G for nbr, nbr2 in itertools.combinations(G[n],2) if nbr in G[nbr2])])

import igraph as ig
import itertools
from typing import Iterable
import random
import pandas as pd
import numpy as np
import scipy.special
import joblib
from randomization import randomize_hyper_df
from utilities import get_df_hyper


import numpy as np
from igraph import *
np.seterr(divide='ignore')



def nodal_eff(g):
    g.es["weight"] = np.ones(g.ecount())
    weights = g.es["weight"][:]
    sp = (1.0 / np.array(g.shortest_paths_dijkstra(weights=weights)))
    np.fill_diagonal(sp,0)
    N=sp.shape[0]
    ne= (1.0/(N-1)) * np.apply_along_axis(sum,0,sp)
    average_distance = np.mean(sp)
    efficiency = np.mean(ne)
    return average_distance,efficiency





def find_triangles(G):
    return set([tuple(sorted(set(triple))) for triple in set(frozenset((n,nbr,nbr2)) for n in G for nbr, nbr2 in itertools.combinations(G[n],2) if nbr in G[nbr2])])



def remove_hubs(df,max_degree):
    df_bi =  get_bipartite_hyper(df)
    board_size = df_bi.groupby('BvD').size()
    busy_board = set(board_size[board_size>max_degree].index)
    df_bi = df_bi[(df['BvD'].isin(busy_board) == False)]
    return from_bipartite_to_hyper_df(df_bi)

def compute_largest_connected_component_fast(hyperlinks, order, largest_cc=None,previous_step_G = ig.Graph()):
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
    ### select only hyperlinks with order == order
    hyperlinks_filt = hyperlinks[hyperlinks['order'] == order].copy()
    ### if the set is empty, return the previous Graph
    if hyperlinks_filt.shape[0] == 0:
        G = previous_step_G
    ### otherwise add the links obtained by decomposing the hyperlinks
    else:
        hyperlinks_filt.loc[:, 'links'] = hyperlinks_filt['BvD'].apply(lambda x: list(itertools.combinations(x, 2)))
        G = previous_step_G
    ### compute the set of hyperlinks obtained from the decomposition of hyperlinks
        G.add_edges(set(itertools.chain.from_iterable(hyperlinks_filt['links'])))
        print ('network created')
    #real nodes in the network are just those with at least a link
    available_nodes = len(G.vs(_degree_gt=0))
    if available_nodes == 0:
        largest_cc = []
        available_nodes = 0
        n_components = 0
        # average_distance = np.nan
        # efficiency = 0 
        n_components = 0
        # diameter = np.nan
    else:
        ### compute the components with igraph
        components = G.components()
        print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

        print('components computed')
        n_components = len(components)
        giant = components.giant()
        largest_cc =giant.vs['name']
        # average_distance,efficiency = nodal_eff(giant)
        # print('efficiency computed')
        # diameter = giant.diameter()
        # print('diameter computed')
    return set(largest_cc), hyperlinks_filt,available_nodes,G,n_components#,average_distance,efficiency,,diameter

def run_resilience_analysis_fast(size_lim=50, revenue_limit='overall', type_data='board', randomization=False, seed1=0, inverse_order=False,unweighted = False,return1 = False,remove_hubs1 = False):
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

    df = df[df['order']>1]
    if remove_hubs1:
        df = remove_hubs(df,50)
        type_data+='_no_hubs'
    print(df.shape)
    dict_nodes_to_id = {node: i for i, node in enumerate(sorted(set(itertools.chain.from_iterable(df['BvD']))))}
    df['BvD'] = df['BvD'].apply(lambda x: tuple([dict_nodes_to_id[node] for node in x]))
    assert ((df['BvD'] == df['BvD'].apply(lambda x:tuple(sorted(x)))).all())
    rand_str = ''
    if type(randomization)!=bool:
        df = randomize_hyper_df(df, seed1=seed1, method=randomization, interlocking_directors=True)
        rand_str = '_rand_'+str(randomization)+'_seed' + str(seed1)

    if unweighted:
        type_data = type_data+ '_unweighted'
    total_hyperlinks = df.drop_duplicates('BvD')
    node_iso = total_hyperlinks.explode(['ISO','BvD']).drop_duplicates('BvD')
    iso_groups = node_iso.groupby('ISO')['BvD'].apply(lambda x:list(x))
    node_iso = node_iso.set_index('BvD')['ISO'].to_dict()
    dict_component = {}
    dict_n_component = {}
    dict_local_component_nodes = {}
    # dict_average_distance = {}
    # dict_efficiency = {}
    # dict_diameter = {}
    iso_connected = {}
    dict_component_nodes = {}
    largest_cc = set()
    order_list = range(2, size_lim+1)[::-1]
    iso_set = set()
    node_set = set()
    inverse_str = ''
    previous_step_G1 = ig.Graph()
    previous_step_G1.add_vertices(range(len(dict_nodes_to_id)))
    
    del dict_nodes_to_id

    if inverse_order: 
        order_list = range(2, size_lim+1)
        inverse_str = '_inverse'
    for order in order_list:
        print(order)
        #largest_cc, hyperlinks_filt, available_nodes,previous_step_G,n_components,average_distance,efficiency,diameter = compute_largest_connected_component_fast(total_hyperlinks, order, largest_cc, previous_step_G=previous_step_G1)
        largest_cc, hyperlinks_filt, available_nodes,previous_step_G,n_components = compute_largest_connected_component_fast(total_hyperlinks, order, largest_cc, previous_step_G=previous_step_G1)

        
        previous_step_G1 = previous_step_G.copy()
        dict_n_component[order] = n_components
        # dict_average_distance[order] = average_distance
        # dict_efficiency[order] = efficiency
        # dict_diameter[order] = diameter
        del previous_step_G
        if largest_cc == set():
            iso_connected[order] = Counter()
            iso_set = set()
            node_set = set()
        else:
            iso_connected[order] = Counter([node_iso[node] for node in list(largest_cc)])
            
        dict_local_component_nodes[order] = Counter(iso_groups.apply(lambda x:previous_step_G1.subgraph(x)).apply(lambda x:len(x.components().giant().vs['name'])).to_dict())

        dict_component[order] = (len(largest_cc), available_nodes)
        dict_component_nodes[order] = largest_cc
        if order == order_list[-1]:
                for order in order_list:
                    assert (len(set(dict_component_nodes[order]) & set(largest_cc)) == dict_component[order][0])
        
        joblib.dump(dict_component, '/data/ceriaac/CORPNET_project/Data/' + type_data_str + '/resilience/resilience_nodes' + str(size_lim) + '_' + revenue_limit + '_' + inverse_str + rand_str + type_data + '_fast_ig.joblib')
        joblib.dump(iso_connected, '/data/ceriaac/CORPNET_project/Data/' + type_data_str + '/resilience/resilience_iso' + str(size_lim) + '_' + revenue_limit + '_' + inverse_str + rand_str + type_data + '_fast_ig.joblib')
        joblib.dump(dict_local_component_nodes, '/data/ceriaac/CORPNET_project/Data/' + type_data_str + '/resilience/resilience_iso_local' + str(size_lim) + '_' + revenue_limit + '_' + inverse_str + rand_str + type_data + '_fast_ig.joblib')

        joblib.dump(dict_n_component, '/data/ceriaac/CORPNET_project/Data/' + type_data_str + '/resilience/resilience_n_comp_' + str(size_lim) + '_' + revenue_limit + '_' + inverse_str + rand_str + type_data + '_fast_ig.joblib')
        # joblib.dump(dict_average_distance, '/data/ceriaac/CORPNET_project/Data/' + type_data_str + '/resilience/avg_distance' + str(size_lim) + '_' + revenue_limit + '_' + inverse_str + rand_str + type_data + '_fast_ig.joblib')
        # joblib.dump(dict_efficiency, '/data/ceriaac/CORPNET_project/Data/' + type_data_str + '/resilience/efficiency' + str(size_lim) + '_' + revenue_limit + '_' + inverse_str + rand_str + type_data + '_fast_ig.joblib')
        # joblib.dump(dict_diameter, '/data/ceriaac/CORPNET_project/Data/' + type_data_str + '/resilience/diameter' + str(size_lim) + '_' + revenue_limit + '_' + inverse_str + rand_str + type_data + '_fast_ig.joblib')

        
        print('resiliance analysis has been done for order ' + str(order))
        
        joblib.dump(dict_component_nodes, '/data/ceriaac/CORPNET_project/Data/' + type_data_str + '/resilience/resilience_node_set' + str(size_lim) + '_' + revenue_limit + '_' + inverse_str + rand_str + type_data + '_fast_ig.joblib')
    
    if return1:
        return dict_component, iso_connected
    else:
        return 0
