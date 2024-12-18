import pandas as pd
from collections import Counter
from itertools import chain
import numpy as np
def get_bipartite_hyper(hyper_df):
    return hyper_df.explode(['nodes','labels'])

def add_same_label_share(df):
    """
    Adds same_label share information to the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to which the same_label share information will be added.

    Returns:
    pandas.DataFrame: The DataFrame with the added same_label share information.
    """
    
    df['same_label_share'] = df['labels'].apply(lambda x:{k:v/len(x) for k,v in Counter(x).items()})
    df['links'] = df['labels'].apply(lambda x:tuple(sorted(set(x))))
    df['max_labels'] = df['same_label_share'].apply(lambda x:max(x, key=x.get))
    df['max_labels_share'] = df['same_label_share'].apply(lambda x:max(x.values()))
    df['same_label_count'] = df['labels'].apply(lambda x:Counter(x))

    return df


def from_bipartite_list_to_hyper_df(bipartite_list, label_list = None, max_order = False, unweighted = False,return_labels_dict = False):
    df_hyper = pd.DataFrame(bipartite_list)
    df_hyper.columns = ['nodes','hyperlink_id']
    df_hyper = df_hyper.groupby('hyperlink_id')['nodes'].apply(lambda x:tuple(sorted(set(x)))).reset_index()
    df_hyper['order'] = df_hyper['nodes'].apply(lambda x:len(x))
    if label_list:
        nodes = sorted(set([x[0] for x in bipartite_list]))
        labels = {k:v for k,v in zip(nodes,label_list)}
        df_hyper['labels'] = df_hyper['nodes'].apply(lambda x:tuple([labels[y] for y in x]))
    if max_order:
        df_hyper = df_hyper[df_hyper['order']<=max_order]
    if unweighted:
        df_hyper = df_hyper.drop_duplicates('nodes')
    if return_labels_dict:
        return df_hyper,labels
    else:
        return df_hyper

def from_hyperlink_list_to_hyper_df(hyperlink_list, label_list = None, max_order = False, unweighted = False,return_labels_dict = False):
    df_hyper = pd.Series(hyperlink_list).reset_index()
    df_hyper.columns = ['hyperlink_id','nodes']
    df_hyper['nodes'] = df_hyper['nodes'].apply(lambda x:tuple(sorted(set(x))))
    df_hyper['order'] = df_hyper['nodes'].apply(lambda x:len(x))
    if label_list:
        nodes = sorted(set(chain.from_iterable(hyperlink_list)))
        labels = {k:v for k,v in zip(nodes,label_list)}
        df_hyper['labels'] = df_hyper['nodes'].apply(lambda x:tuple([labels[y] for y in x]))
    if max_order:
        df_hyper = df_hyper[df_hyper['order']<=max_order]
    if unweighted:
        df_hyper = df_hyper.drop_duplicates('nodes')
    if return_labels_dict:
        return df_hyper,labels
    else:
        return df_hyper
    

def get_df_hyper(link_list,label_list=False, size_lim=False,unweighted = False,return_labels_dict = False):
    """
    Load and preprocess the DataFrame from a CSV file.

    Args:
        link_list (list): Either a list of links of a bipartite network (node,hyperlink_id) tuples or a list of hyperlinks (v1,...,vn).
        label_list (list, optional): A list of labels for the nodes, ordered by the node index.
        size_lim (int, optional): The maximum order of hyperlinks to consider. Defaults to False.
        unweighted (bool, optional): Whether to consider the hypergraph as unweighted. Defaults to False.
    Returns:
        pandas.DataFrame: The preprocessed DataFrame with columns 'nodes', 'hyperlink_id', 'order', and 'labels'(if available).
    """
    # Read the CSV file into a DataFrame
    try:
        df = from_bipartite_list_to_hyper_df(link_list, label_list, max_order = size_lim, unweighted = unweighted,return_labels_dict = return_labels_dict)
    except:
        df = from_hyperlink_list_to_hyper_df(link_list, label_list, max_order = size_lim, return_labels_dict = return_labels_dict)
    if return_labels_dict:
        df,labels = df
    df = df[df['order'] > 1]
    if return_labels_dict:
        return df,labels
    else: 
        return df

def from_bipartite_to_hyper_df(df):
    """
    Convert a bipartite DataFrame to a hypergraph DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame representing a bipartite graph.

    Returns:
        pandas.DataFrame: The resulting DataFrame representing a hypergraph.
    """
    # Create a new DataFrame with index as unique full names
    df_hyper = pd.DataFrame(index=sorted(df.hyperlink_id.unique()))

    # Group the DataFrame by full name and get the unique sorted set of nodes values for each group
    hyper_companies = df.groupby('hyperlink_id')['nodes'].apply(lambda x: tuple(sorted(set(x))))

    # Group the DataFrame by full name and get the unique sorted set of labels values for each group
    hyper_countries = df.groupby('hyperlink_id')['labels'].apply(lambda x: tuple(sorted((x))))

    # Assign the hyper_companies and hyper_countries values to the corresponding columns in df_hyper
    df_hyper['nodes'] = hyper_companies
    df_hyper['labels'] = hyper_countries 

    # Calculate the order of each hypergraph by counting the number of nodes values in each group
    df_hyper['order'] = hyper_companies.apply(lambda x: len(x))
    
    
    # Return the resulting hypergraph DataFrame
    return df_hyper


def get_order_relevance(sr):
    return (np.trapz(y = sr.values,x = sr.index)-0.5)/(max(sr.index)-min(sr.index)-0.5)


def get_complementary_order_relevance(sr,max_order):
    sr.sort_index(inplace = True,ascending = False)
    sr = sr.loc[max_order:]
    sr.loc[max_order +1] = 0
    sr.index = sr.index - 1
    
    return max(sr.values)-sr.sort_index().loc[2:]


def compute_average_order_relevance(measure,complementary_measure):
    return 0.5 * (measure + complementary_measure)
    
def compute_order_gap(measure,complementary_measure):
    return measure - complementary_measure
