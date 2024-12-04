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
from utilities import get_df_hyper
import sys,gc
import psutil
from psutil._common import bytes2human
from collections import Counter
def find_triangles(G):
    return set([tuple(sorted(set(triple))) for triple in set(frozenset((n,nbr,nbr2)) for n in G for nbr, nbr2 in itertools.combinations(G[n],2) if nbr in G[nbr2])])






def compute_clustering_coefficient_fast(hyperlinks, order, compute_triangles = False,previous_step_triangles = set(),previous_step_G = nx.Graph(),inverse_order = False):
    """
    Computes the clustering coefficient in a graph of hyperlinks.

    Args:
        hyperlinks (pandas.DataFrame): DataFrame containing the hyperlinks data.
        order (int): Maximum order of hyperlinks to consider.
        inverse_order (bool): if False, include all hyperlinks with order greater than or equal to the given order. If True, include all hyperlinks with order smaller than or equal to the given order.

    Returns:
        tuple: A tuple containing the average clustering coefficient and the global clustering coefficient.
    """
    
    hyperlinks_filt = hyperlinks[hyperlinks['order'] == order]

    if hyperlinks_filt.shape[0] == 0:
        closed_triangles = previous_step_triangles
        G = previous_step_G
    else:
        if compute_triangles:
            if order>=3:
                closed_triangles = previous_step_triangles.union(*hyperlinks[hyperlinks['order'] == order]['BvD'].apply(lambda x: set(itertools.combinations(x,3))).values)
            else:
                closed_triangles = previous_step_triangles
        else:
            closed_triangles = previous_step_triangles

        print('closed triangles computed')

        
        hyperlinks_filt = hyperlinks_filt['BvD'].apply(lambda x: list(itertools.combinations(x, 2))).values

        print(str(order)+' hyperlinks_filtered')
        G = previous_step_G
        G.add_edges_from(set(itertools.chain.from_iterable(hyperlinks_filt)))
        print('network created')
    if len(G.nodes) == 0:
        clustering_coeff_avg = 0
        clustering_coeff_global = 0
        open_triangles,closed_triangles = 0,previous_step_triangles
        n_triangles = 0
        contri = 0

    else:
        clustering_coeff_global = 0
        print('transitivity computed')

        clustering_coeff_avg = 0
        print('average clustering computed')
        n_triangles = sum(nx.triangles(G).values()) / 3
        open_triangles = n_triangles - len(closed_triangles)

    return clustering_coeff_avg,clustering_coeff_global,open_triangles,closed_triangles,n_triangles,G
    
    
def run_clustering_analysis_fast(size_lim=50, revenue_limit='overall', type_data='board', randomization=False, seed1=0, inverse_order=False,unweighted = False,compute_triangles = False,save = True,opt = True):
    """
    Run clustering analysis on the given dataset.

    Parameters:
    - size_lim (int): The size limit for the analysis.
    - revenue_limit (str): The revenue limit for the analysis.
    - type_data (str): The type of data to be used for the analysis.
    - randomization (bool): Whether to perform randomization on the dataset.
    - seed1 (int): The seed value for randomization.
    - inverse_order (bool): if False, include all hyperlinks with order less than or equal to the given order. If True, include all hyperlinks with order greater than or equal to the given order.

    Returns:
    - dict_average_clustering (dict): A dictionary containing the average clustering values for each order.
    - dict_global_clustering (dict): A dictionary containing the global clustering values for each order.
    """
    from tqdm import tqdm
    from time import sleep
    import psutil

    process = process_memory()
    
        
    if type_data == 'board':
        type_data_str = 'inconsistent_ISO'
    else:
        type_data_str = type_data
    if opt: opt_str = '_opt'
    else: opt_str = ''

    df = get_df_hyper(size_lim, revenue_limit, type_data=type_data,unweighted=unweighted)
    df = df[df['order']>1]
    mem0 = process_memory()

    rand_str = ''
    if unweighted:
        type_data = type_data+ '_unweighted'

    if randomization:
        df = randomize_hyper_df(df, seed1=seed1, method=randomization, interlocking_directors=True)
        rand_str = '_rand_'+str(randomization)+'_seed' + str(seed1)
    print(bytes2human(process_memory() - mem0))

    total_hyperlinks = df
    total_hyperlinks.drop_duplicates('BvD',inplace = True)
    dict_average_clustering = {}
    dict_global_clustering = {}
    dict_open_triangles = {}
    dict_n_triangles = {}
    degree_sum = {}
    previous_step_triangles1 = set()
    previous_step_G1 = nx.Graph()
    order_list = range(2, size_lim+1)
    inverse_str = ''
    print(bytes2human(process_memory() - mem0))

    if inverse_order:
        order_list = order_list[::-1]
        inverse_str = '_inverse'
    for order in order_list:
        avg_clustering, global_clustering,open_triangles, previous_step_triangles,n_triangles,previous_step_G = compute_clustering_coefficient_fast(total_hyperlinks, order, compute_triangles=compute_triangles,previous_step_triangles=previous_step_triangles1,previous_step_G = previous_step_G1)
        if opt:
            previous_step_G1 = previous_step_G
            previous_step_triangles1 = previous_step_triangles
        else:
            previous_step_G1 = previous_step_G.copy()
            previous_step_triangles1 = previous_step_triangles.copy()

        
        del previous_step_triangles,previous_step_G
        # gc.collect()
        dict_average_clustering[order] = avg_clustering
        dict_global_clustering[order] = global_clustering
        dict_open_triangles[order] = open_triangles
        dict_n_triangles[order] = n_triangles
        if save:

            if compute_triangles:
                pd.Series(dict_open_triangles).to_csv('/data/ceriaac/CORPNET_project/Data/' + type_data_str + '/clustering/open_triangles' + str(size_lim) + '_' + revenue_limit + '_' + inverse_str + rand_str + type_data + opt_str + '_fast.gz',compression = 'gzip')
                print('joblib file3 saved')
                
            pd.Series(dict_n_triangles).to_csv('/data/ceriaac/CORPNET_project/Data/' + type_data_str + '/clustering/n_triangles' + str(size_lim) + '_' + revenue_limit + '_' + inverse_str + rand_str + type_data + opt_str+'_fast.gz',compression = 'gzip')
            
            
            print('joblib file3 saved')
            
        print('resiliance analysis has been done for order ' + str(order))
        
        print(bytes2human(process_memory() - mem0))
    del total_hyperlinks,previous_step_G1,df
    
    return None


def compute_largest_connected_component_fast(hyperlinks, order, largest_cc=None,inverse_order = False,previous_step_G = nx.Graph()):
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
        
    hyperlinks_filt = hyperlinks[hyperlinks['order'] == order].copy()
    if hyperlinks_filt.shape[0] == 0:
        
        G = previous_step_G
    else:
        hyperlinks_filt.loc[:, 'links'] = hyperlinks_filt['BvD'].apply(lambda x: list(itertools.combinations(x, 2)))
        G = previous_step_G
        G.add_edges_from(hyperlinks_filt['links'].explode('links').unique())
    available_nodes = len(G.nodes)
    if available_nodes == 0:
        largest_cc = []
        available_nodes = 0
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        
    return set(largest_cc), hyperlinks_filt,available_nodes,G

def run_resilience_analysis_fast(size_lim=50, revenue_limit='overall', type_data='board', randomization=False, seed1=0, inverse_order=False,unweighted = False):
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
    rand_str = ''
    if randomization:
        df = randomize_hyper_df(df, seed1=seed1, method=randomization, interlocking_directors=True)
        rand_str = '_rand_' + str(seed1)

    if unweighted:
        type_data = type_data+ '_unweighted'
        
    total_hyperlinks = df
    total_hyperlinks.drop_duplicates('BvD',inplace=True)
    dict_component = {}
    iso_connected = {}
    dict_component_nodes = {}
    largest_cc = set()
    order_list = range(2, size_lim+1)[::-1]
    inverse_str = ''
    previous_step_G1 = nx.Graph()
    iso_set = set()
    if inverse_order: 
        order_list = range(2, size_lim+1)
        inverse_str = '_inverse'
    for order in order_list:
        print(order)
        largest_cc, hyperlinks_filt, available_nodes,previous_step_G = compute_largest_connected_component_fast(total_hyperlinks, order, largest_cc, inverse_order=inverse_order,previous_step_G=previous_step_G1)
        previous_step_G1 = previous_step_G.copy()
        del previous_step_G
        if largest_cc == set():
            iso_connected[order] = 0
            iso_set = set()
        else:
            if hyperlinks_filt.shape[0] == 0:
                iso_connected[order] = iso_set
            else:
                iso_connected[order] = iso_set.union(hyperlinks_filt[(hyperlinks_filt['BvD'].apply(lambda x: len(set(x) & largest_cc) > 0))].explode('ISO')['ISO'].unique())
        dict_component[order] = (len(largest_cc), available_nodes)
        dict_component_nodes[order] = largest_cc
        joblib.dump(dict_component, '/data/ceriaac/CORPNET_project/Data/' + type_data_str + '/resilience/resilience_nodes' + str(size_lim) + '_' + revenue_limit + '_' + inverse_str + rand_str + type_data + '_fast.joblib')
        joblib.dump(iso_connected, '/data/ceriaac/CORPNET_project/Data/' + type_data_str + '/resilience/resilience_iso' + str(size_lim) + '_' + revenue_limit + '_' + inverse_str + rand_str + type_data + '_fast.joblib')
        print('resiliance analysis has been done for order ' + str(order))
        
    #joblib.dump(dict_component_nodes, '/data/ceriaac/CORPNET_project/Data/' + type_data_str + '/resilience/resilience_node_set' + str(size_lim) + '_' + revenue_limit + '_' + inverse_str + rand_str + type_data + '.joblib')
    del total_hyperlinks,largest_cc
    return dict_component, iso_connected
