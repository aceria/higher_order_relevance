from utilities import get_df_hyper,add_same_label_share,get_order_relevance
import numpy as np
import itertools
import pandas as pd
from scipy.special import binom
from collections import Counter

dict_str_to_limit = {k:v for k,v in zip(['1M','5M','10M','50M'],[1e6,5e6,10e6,50e6])}



# Function to calculate order contribution of link weights based on given dataframe and order range
def get_order_contribution(df, order_range=range(2, 51), binned=False):
    # Calculate the sum of weights for each order
    sr_contribution = df.groupby('order')['weights'].apply(sum).fillna(0)
    
    # Check for missing orders in the given range and assign 0 weight to them
    missing_orders = set(order_range) - set(sr_contribution.index)
    for i in missing_orders:
        sr_contribution.loc[i] = 0
    
    # If binned is True, group the orders into bins and calculate the sum of weights for each bin
    if binned:
        df = sr_contribution.reset_index()
        df.columns = ['order', 'weights']
        df['bins_size'] = pd.cut(df['order'], np.logspace(np.log10(min(order_range) - 1e-3), np.log10(max(order_range) + 1e-3), 50)).apply(lambda x: x.mid)
        sr_contribution = df.groupby('bins_size')['weights'].apply(sum).fillna(0)
    
    # Reset the index and rename the columns
    sr_contribution = sr_contribution.reset_index()
    sr_contribution.columns = ['order', 'weights']
    
    # Sort the dataframe by order and calculate the normalized weights
    sr_contribution.sort_values('order', inplace=True)
    sr_contribution['norm_weights'] = (sr_contribution['weights'] / sr_contribution['weights'].sum()).cumsum()
    
    return sr_contribution.sort_values('order')



def get_trans_hyper_df(df):
    """
    Returns a modified DataFrame with the inter-label order contribution to link weights for each hyperlinks.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: The transformed DataFrame.
    """
    # Filter the dataframe to include only different label hyperlinks
    trans_hyper = df[df['max_labels_share'] < 1].copy()
    
    # Calculate the link contribution for each different label hyperlink, as the product of the number of nodes in each pair of labels connected by the hyperlink
    trans_hyper['link_contrib'] = trans_hyper['labels'].apply(lambda x: Counter(x)).apply(lambda x: {(k1, k2): x[k1] * x[k2] for k1, k2 in itertools.combinations(x.keys(), 2)})
    
    # Explode the link_contrib column into multiple rows (each row will be an labels and its weight)
    trans_hyper = trans_hyper.explode('link_contrib')
    
    # Extract the labels and weights from the link_contrib column
    trans_hyper['labels'] = trans_hyper['link_contrib'].apply(lambda x: x[0])
    trans_hyper['weights'] = trans_hyper['link_contrib'].apply(lambda x: x[1])
    
    return trans_hyper

def get_internal_hyper(df):
    """
    Returns a modified DataFrame with the intra-label order contribution for each hyperlinks.

    

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: A modified DataFrame with internal hyperlinks.
    """

    internal_hyper = df.copy()

    # Compute the number of nodes based in each labels in each hyperlink
    internal_hyper['labels_counter'] = internal_hyper["labels"].apply(lambda x:Counter(x))

    # Compute the number of different countries in each hyperlink
    internal_hyper['labels'] = internal_hyper['labels_counter'].apply(lambda x:list(x.keys()))

     # Separate the contribution of each labels in each hyperlink
    internal_hyper = internal_hyper.explode('labels')

    # Calculate the link contribution for each hyperlink, for each labels, this is the binomial coefficient of the number of nodes in the hyperlink in each labels
    internal_hyper['weights'] = internal_hyper.apply(lambda x:binom(x.labels_counter[x.labels],2),axis = 1)

    return internal_hyper



#### Routines to compute the order relevance


def compute_order_relevance(df1):
    df1 = df1.set_index('order')['norm_weights'].sort_index()
    return get_order_relevance(df1)
    


def compute_order_relevance_country (df_order_rel,type_analysis = 'same_label'):
   

    border_order_relevance = {}

    for (labels,border), df in df_order_rel.groupby(['labels','border']):
        border_order_relevance[(labels,border)] = (compute_order_relevance(df),max(df['order']),min(df['order']))
    
    
    a1 = [border_order_relevance[k][0] for k in border_order_relevance if k[1] == type_analysis],[border_order_relevance[k][1] for k in border_order_relevance if k[1] == type_analysis],[k[0] for k in border_order_relevance if k[1] == type_analysis],[border_order_relevance[k][2] for k in border_order_relevance if k[1] == type_analysis]
    df_order = pd.DataFrame(a1).T.set_index(2)
    df_order.columns = ['order_relevance','max_order','min_order']
    
    return df_order

def run_order_relevance_analysis(links_list,labels_list,size_lim=50, type_analysis='overall', binned=False, unweighted=False,save = False):
    """
    Run the order relevance analysis on the given dataset.

    Parameters:
    size_lim (int): The size limit for the dataset.
    type_analysis (str): The type of analysis to perform ('labelled' or 'overall'). If "overall", it computes the order relevance for the overall dataset, if "labelled" it compute the same label and different label order contribution to link weights.
    binned (bool): Whether to group the orders into bins.
    unweighted (bool): Whether to perform the analysis on unweighted data.

    Returns:
    if type_analysis == 'overall': a tuple (pandas.DataFrame,float): The order contribution and the order relevance of the given higher-order network analysis results.
    if type_analysis == 'labelled': a tuple(pandas.DataFrame1, (pandas.DataFrame2,pandas.DataFrame3)): DataFrame1 is the dataframe with the inter and intra-label order contribution of each label, while DataFrame2 (DataFrame3) is the dataframe with the intra-(inter-)order relevance of each label.
    """

    sr_list = list()
    df = get_df_hyper(links_list, labels_list, size_lim=size_lim, unweighted=unweighted)
    df = add_same_label_share(df)
    print('data loaded')

    if type_analysis == 'labelled':
        df_tot_trans = get_trans_hyper_df(df)
        df_tot_internal = get_internal_hyper(df)
        order_range = range(min([df_tot_trans['order'].min(), df_tot_internal['order'].min()]), max([df_tot_trans['order'].max(), df_tot_internal['order'].max()]) + 1)

        # Calculate order relevance for each labels and border type
        for labels, df_rest in df_tot_internal.groupby('labels'):
            sr = get_order_contribution(df_rest, order_range=range(2, max(df_rest['order'] + 1)), binned=binned)
            sr = sr.reset_index()
            sr['labels'] = labels
            sr['border'] = 'same_label'

            sr_list.append(sr)

        for labels, df in df_tot_trans.explode('labels').groupby('labels'):
            sr = get_order_contribution(df, order_range=range(2, max(df['order']) + 1), binned=binned)
            sr = sr.reset_index()
            sr['labels'] = labels
            sr['border'] = 'different_label'
            sr_list.append(sr)
        sr = pd.concat(sr_list)
    
    if type_analysis == 'overall':
        order_range = range(min(df['order']), max(df['order']) + 1)
        df['weights'] = df['order'].apply(lambda x: binom(x, 2))
        sr = get_order_contribution(df, order_range=order_range, binned=binned)
        sr = sr.reset_index()
        sr['labels'] = 'overall'
    
    unweighted_str = ''
    if unweighted:
        unweighted_str = '_unw'
    
    if type_analysis == 'overall':
        order_relevance = compute_order_relevance(sr)
        sr = (sr[['order','norm_weights']].set_index('order')['norm_weights'].sort_index(),order_relevance)
    else:
        order_relevance_same_label = compute_order_relevance_country(sr,type_analysis = 'same_label')
        order_relevance_different_label = compute_order_relevance_country(sr,type_analysis = 'different_label')
        dict1 = {}
        for (border,label), df in sr.groupby(['border','labels']):
            print(border)
            dict1[f'{border}_{label}'] = df.set_index('order')['norm_weights'].sort_index()
        sr = {'order_contribution':dict1,'order_relevance':{'same_label':order_relevance_same_label,'different_label':order_relevance_different_label}}

     # Save the order relevance analysis results to a CSV file
    if save: sr.to_csv('../Data/order_relevance/order_relevance_' + str(size_lim) + '_' + '_' + type_analysis + unweighted_str + '.csv')

    return sr

