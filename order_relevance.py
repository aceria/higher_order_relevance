from utilities import get_df_hyper,add_national_share
import numpy as np
import itertools
import pandas as pd
from scipy.special import binom
from collections import Counter

dict_str_to_limit = {k:v for k,v in zip(['1M','5M','10M','50M'],[1e6,5e6,10e6,50e6])}



# Function to calculate order relevance of link weights based on given dataframe and order range
def get_order_relevance(df, order_range=range(2, 51), binned=False):
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
    # Filter the dataframe to include only international hyperlinks
    trans_hyper = df[df['max_country_share'] < 1].copy()
    
    # Calculate the link contribution for each international hyperlink, as the product of the number of nodes in each pair of country connected by the hyperlink
    trans_hyper['link_contrib'] = trans_hyper['ISO'].apply(lambda x: Counter(x)).apply(lambda x: {(k1, k2): x[k1] * x[k2] for k1, k2 in itertools.combinations(x.keys(), 2)})
    
    # Explode the link_contrib column into multiple rows (each row will be an iso and its weight)
    trans_hyper = trans_hyper.explode('link_contrib')
    
    # Extract the ISO and weights from the link_contrib column 
    trans_hyper['ISO'] = trans_hyper['link_contrib'].apply(lambda x: x[0])
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

    # Compute the number of nodes based in each country in each hyperlink
    internal_hyper['ISO_counter'] = internal_hyper["ISO"].apply(lambda x:Counter(x))

    # Compute the number of different countries in each hyperlink
    internal_hyper['ISO'] = internal_hyper['ISO_counter'].apply(lambda x:list(x.keys()))

     # Separate the contribution of each country in each hyperlink
    internal_hyper = internal_hyper.explode('ISO')

    # Calculate the link contribution for each hyperlink, for each country, this is the binomial coefficient of the number of nodes in the hyperlink in each country
    internal_hyper['weights'] = internal_hyper.apply(lambda x:binom(x.ISO_counter[x.ISO],2),axis = 1)

    return internal_hyper

def run_order_relevance_analysis(size_lim=50, revenue_limit='overall', type_data='board', type_analysis='overall', binned=False, unweighted=False):
    """
    Run the order relevance analysis on the given dataset.

    Parameters:
    size_lim (int): The size limit for the dataset.
    revenue_limit (str): The revenue limit for the dataset.
    type_data (str): The type of data to analyze ('board' or 'collaboration_max_iso').
    type_analysis (str): The type of analysis to perform ('labelled' or 'overall'). If "overall", it computes the order relevance for the overall dataset, if "labelled" it compute the national and international order contribution to link weights.
    binned (bool): Whether to group the orders into bins.
    unweighted (bool): Whether to perform the analysis on unweighted data.

    Returns:
    pandas.DataFrame: The order relevance analysis results.
    """
    if type_data == 'board':
        file_path = 'inconsistent_ISO'
    else:
        file_path = type_data
    sr_list = list()
    df = get_df_hyper(size_lim, revenue_limit, type_data=type_data)
    df = df[df['order'] > 1]
    if unweighted:
        df.drop_duplicates('BvD', inplace=True)
    df = add_national_share(df)
    print('data loaded')

    if type_analysis == 'labelled':
        df_tot_trans = get_trans_hyper_df(df)
        df_tot_internal = get_internal_hyper(df)
        order_range = range(min([df_tot_trans['order'].min(), df_tot_internal['order'].min()]), max([df_tot_trans['order'].max(), df_tot_internal['order'].max()]) + 1)

        # Calculate order relevance for each country and border type
        for iso, df_rest in df_tot_internal.groupby('ISO'):
            sr = get_order_relevance(df_rest, order_range=range(2, max(df_rest['order'] + 1)), binned=binned)
            sr = sr.reset_index()
            sr['iso'] = iso
            sr['border'] = 'national'

            sr_list.append(sr)

        for iso, df in df_tot_trans.explode('ISO').groupby('ISO'):
            sr = get_order_relevance(df, order_range=range(2, max(df['order'] + 1)), binned=binned)
            sr = sr.reset_index()
            sr['iso'] = iso
            sr['border'] = 'international'
            sr_list.append(sr)
        sr = pd.concat(sr_list)
    
    if type_analysis == 'overall':
        order_range = range(2, 51)
        df['weights'] = df['order'].apply(lambda x: binom(x, 2))
        sr = get_order_relevance(df, order_range=range(2, max(df['order'] + 1)), binned=binned)
        sr = sr.reset_index()
        sr['iso'] = 'overall'
        sr['border'] = 'overall'
    
    unweighted_str = ''
    if unweighted:
        unweighted_str = '_unw'
    
    # Save the order relevance analysis results to a CSV file
    sr.to_csv('/data/ceriaac/CORPNET_project/Data/' + file_path + '/order_relevance/order_relevance_' + str(size_lim) + '_' + revenue_limit + '_' + type_analysis + unweighted_str + '.csv')
    
    return sr
