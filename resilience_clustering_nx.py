import networkx as nx
import itertools
import pandas as pd
import numpy as np
import joblib
from randomization import randomize_hyper_df
from utilities import get_df_hyper, get_complementary_order_relevance, get_order_relevance
from collections import Counter
def find_triangles(G):
    return set([tuple(sorted(set(triple))) for triple in set(frozenset((n,nbr,nbr2)) for n in G for nbr, nbr2 in itertools.combinations(G[n],2) if nbr in G[nbr2])])








def compute_clustering_coefficient_fast(hyperlinks, order,previous_step_G = nx.Graph(),verbose = False):
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
        G = previous_step_G
    else:
        
        hyperlinks_filt = hyperlinks_filt['nodes'].apply(lambda x: list(itertools.combinations(x, 2))).values

        if verbose: print(str(order)+' hyperlinks_filtered')
        G = previous_step_G
        G.add_edges_from(set(itertools.chain.from_iterable(hyperlinks_filt)))
        if verbose: print('network created')
    if len(G.nodes) == 0:
        n_triangles = 0

    else:
        if verbose: print('transitivity computed')

        if verbose: print('average clustering computed')
        n_triangles = sum(nx.triangles(G).values()) / 3

    return n_triangles,G
    
    
def run_clustering_analysis_fast(link_list, label_list=False, size_lim=50, randomization=False, seed1=0, inverse_order=False, unweighted=False, save=True, verbose=False):
    """
    Compute the order contribution and relevance to the number of triangles.

    Args:
        links_list (list): A list of links of a bipartite network (node, item) or a list of hyperlinks .
        labels_list (list,optional): A list of labels, the index of the label should correspond to node id. Defaults to False.
        size_lim (int, optional): Size limit for the analysis. Defaults to 50.
        randomization (bool, optional): Whether to perform randomization. Defaults to False.
        seed1 (int, optional): Seed value for randomization. Defaults to 0.
        inverse_order (bool, optional): Whether to use inverse order. Defaults to False.
        unweighted (bool, optional): Whether to use unweighted data. Defaults to False.
        save (bool, optional): Whether to save the results. Defaults to True.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        dict: A dictionary containing the order contribution, order relevance, and overall value of the analysis.
    """    
    df = get_df_hyper(link_list, label_list, size_lim=size_lim, unweighted=unweighted) 
    if size_lim > df['order'].max():
        size_lim = df['order'].max()
    df = df[df['order']>1]
    rand_str = ''
    type_data = ''
    if unweighted:
        type_data = type_data+ '_unweighted'

    if randomization:
        df = randomize_hyper_df(df, seed1=seed1, method=randomization, interlocking_directors=True)
        rand_str = '_rand_'+str(randomization)+'_seed' + str(seed1)

    total_hyperlinks = df
    total_hyperlinks.drop_duplicates('nodes',inplace = True)
    dict_n_triangles = {}
    previous_step_G1 = nx.Graph()
    order_list = range(2, size_lim+1)
    inverse_str = ''

    if inverse_order:
        order_list = order_list[::-1]
        inverse_str = '_inverse'
    for order in order_list:
        n_triangles,previous_step_G = compute_clustering_coefficient_fast(total_hyperlinks, order, previous_step_G = previous_step_G1)
        previous_step_G1 = previous_step_G
        dict_n_triangles[order] = n_triangles
        
        if verbose: print('resiliance analysis has been done for order ' + str(order))
    del total_hyperlinks,previous_step_G1,df

    order_contribution = pd.Series(dict_n_triangles)
    if inverse_order:
        order_contribution = get_complementary_order_relevance(order_contribution,size_lim)
    order_relevance = get_order_relevance(order_contribution/order_contribution.max())
    overall_value = order_contribution.max()
    order_contribution = order_contribution/overall_value
    result1 = {'order_contribution':order_contribution,'order_relevance':order_relevance, 'overall_value':overall_value}

    if save:

                    
                joblib.dump(result1,'Data/n_triangles' + str(size_lim) + '_' + inverse_str + rand_str + type_data +'_fast.joblib')
                
                
                if verbose: print('joblib file3 saved')
                
            
    
    return result1


def compute_largest_connected_component_fast(hyperlinks, order, largest_cc=None,previous_step_G = nx.Graph()):
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
        hyperlinks_filt.loc[:, 'links'] = hyperlinks_filt['nodes'].apply(lambda x: list(itertools.combinations(x, 2)))
        G = previous_step_G
        G.add_edges_from(hyperlinks_filt['links'].explode('links').unique())
    available_nodes = len(G.nodes)
    if available_nodes == 0:
        largest_cc = []
        available_nodes = 0
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        
    return set(largest_cc), hyperlinks_filt,available_nodes,G

def run_resilience_analysis_fast(link_list, label_list, size_lim=50, randomization=False, seed1=0, inverse_order=False, unweighted=False, verbose=False, save=False, size_lim_local=True):
    """
    Compute the order contribution and relevance to the size of the largest connected component and to the number of nodes of each label in the largest connected component.

    Parameters:
    -links_list (list): A list of links of a bipartite network (node, item) or a list of hyperlinks .
    -labels_list (list,optional): A list of labels, the index of the label should correspond to node id.
    - size_lim (int, optional): Maximum size limit for the analysis. Defaults to 50.
    - randomization (bool, optional): Whether to perform randomization. Defaults to False.
    - seed1 (int, optional): Seed value for randomization. Defaults to 0.
    - inverse_order (bool, optional): Whether to perform analysis in inverse order. Defaults to False.
    - unweighted (bool, optional): Whether to consider the network as unweighted. Defaults to False.
    - verbose (bool, optional): Whether to print verbose output. Defaults to False.
    - save (bool, optional): Whether to save the results. Defaults to False.
    - size_lim_local (bool, optional): Whether to consider local size limit. Defaults to True.

    Returns:
    - dict: Dictionary containing the results of the resilience analysis.

    """
    

    df,labels_dict = get_df_hyper(link_list, label_list, size_lim=size_lim, unweighted=unweighted,return_labels_dict=True)    
    if size_lim > df['order'].max():
        size_lim = df['order'].max()
    df = df[df['order']>1] 
    rand_str = ''
    if randomization:
        df = randomize_hyper_df(df, seed1=seed1, method=randomization, interlocking_directors=True)
        rand_str = '_rand_' + str(seed1)
    type_data = ''
    if unweighted:
        type_data = type_data+ '_unweighted'
    if size_lim_local:
        type_data = type_data+ '_local'
    total_hyperlinks = df
    total_hyperlinks.drop_duplicates('nodes',inplace=True)
    dict_component = {}
    labels_connected = {}
    dict_component_nodes = {}
    largest_cc = set()
    order_list = range(2, size_lim+1)
    inverse_str = ''
    previous_step_G1 = nx.Graph()
    labels_set = set()
    if inverse_order: 
        order_list = range(2, size_lim+1)[::-1]
        inverse_str = '_inverse'
    for order in order_list:
        print(order)
        largest_cc, hyperlinks_filt, available_nodes,previous_step_G = compute_largest_connected_component_fast(total_hyperlinks, order, largest_cc, previous_step_G=previous_step_G1)
        previous_step_G1 = previous_step_G.copy()
        del previous_step_G
        if largest_cc == set():
            labels_connected[order] = 0
            labels_set = set()
        else:
            if hyperlinks_filt.shape[0] == 0:
                labels_connected[order] = labels_set
            else:
                labels_connected[order] = labels_set.union(hyperlinks_filt[(hyperlinks_filt['nodes'].apply(lambda x: len(set(x) & largest_cc) > 0))].explode('labels')['labels'].unique())
        dict_component[order] = len(largest_cc)
        labels_connected[order] = Counter([labels_dict[node] for node in list(largest_cc)])
        #### overall analysis
        
        

        
        if verbose: print('resiliance analysis has been done for order ' + str(order))

    order_contribution = pd.Series(dict_component).sort_index()
    if inverse_order:
        order_contribution = get_complementary_order_relevance(order_contribution,size_lim)
    order_relevance = get_order_relevance(order_contribution/order_contribution.max())
    overall_value = order_contribution.max()
    results_overall = {'order_contribution':order_contribution/overall_value,'order_relevance':order_relevance, 'overall_value':overall_value}


    local_order_contribution = {}
    local_order_relevance = {}  
    local_overall_values = {}
    for label in sorted(set(labels_dict.values())):
        local_order_contribution[label] = pd.Series({order:labels_connected[order][label] for order in labels_connected.keys()}).sort_index()
        if inverse_order:
            local_order_contribution[label] = get_complementary_order_relevance(local_order_contribution[label],size_lim)
        if size_lim_local:
            max_order = min(local_order_contribution[label][local_order_contribution[label]==max(local_order_contribution[label])].index)
        else:
            max_order = size_lim
        local_order_contribution[label] = local_order_contribution[label].loc[:max_order]
        local_order_contribution[label]
        local_order_relevance[label] = get_order_relevance(local_order_contribution[label]/local_order_contribution[label].max())
        local_overall_values[label] = local_order_contribution[label].max()
        local_order_contribution[label] = local_order_contribution[label]/local_overall_values[label]

    results_local = {'order_contribution':local_order_contribution, 'order_relevance':local_order_relevance, 'overall_value':local_overall_values}
    results = {'overall':results_overall,'local':results_local}
    if save:
        joblib.dump(results, 'Data/' + 'resilience_results' + str(size_lim) + '_' + inverse_str + rand_str + type_data + '.joblib')
    del total_hyperlinks,largest_cc
    return results
