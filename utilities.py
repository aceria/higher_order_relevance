import pandas as pd
from collections import Counter
import networkx as nx
from collections.abc import Iterable


def get_bipartite_hyper(hyper_df):
    return hyper_df.explode(['BvD','ISO'])

def add_national_share(df):
    """
    Adds national share information to the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to which the national share information will be added.

    Returns:
    pandas.DataFrame: The DataFrame with the added national share information.
    """
    
    df['national_share'] = df['ISO'].apply(lambda x:{k:v/len(x) for k,v in Counter(x).items()})
    df['links'] = df['ISO'].apply(lambda x:tuple(sorted(set(x))))
    df['max_country'] = df['national_share'].apply(lambda x:max(x, key=x.get))
    df['max_country_share'] = df['national_share'].apply(lambda x:max(x.values()))
    df['national_count'] = df['ISO'].apply(lambda x:Counter(x))

    return df


def get_df_hyper(size_lim=50, revenue_limit='overall', type_data = 'board',unweighted = False):
    """
    Load and preprocess the DataFrame from a CSV file.

    Args:
        size_lim (int, optional): The size limit for the DataFrame. Defaults to 50.
        revenue_limit (str, optional): The revenue limit for the DataFrame. Defaults to 'overall'.
        type_data (str, optional): The type of data to load. Defaults to 'board'.
    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
    # Read the CSV file into a DataFrame
   
       
    if type_data == 'board':
         if revenue_limit == 'overall':df = pd.read_csv('/data/ceriaac/CORPNET_project/Data/Preprocessed/all_position_static_keep_inconsistent_ISO/Numbered/higher_order_com_nodes/all_positions_static_clean_'+str(size_lim)+'_'+revenue_limit+'_higher_order.gz')
         else: df = pd.read_csv('/data/ceriaac/CORPNET_project/Data/Preprocessed/all_position_static_keep_inconsistent_ISO/Numbered/higher_order_com_nodes/all_positions_static_clean_overall_'+revenue_limit+'_higher_order.gz')
         df = df[df['order']<=size_lim]
    elif type_data == 'collaboration_diff_iso':
        df = pd.read_csv('/data/ceriaac/CORPNET_project/Data/Preprocessed/collaboration/Numbered/higher_order/collaboration_iso_diff_static_clean_overall_overall_higher_order.gz')

    elif type_data == 'collaboration_max_iso':
        df = pd.read_csv('/data/ceriaac/CORPNET_project/Data/Preprocessed/collaboration/Numbered/higher_order/collaboration_iso_max_static_clean_overall_overall_higher_order.gz')

    # Convert the 'ISO' and 'BvD' columns from string representation to tuples
    df['ISO'] = df['ISO'].apply(lambda x: tuple(eval(x)))
    df['BvD'] = df['BvD'].apply(lambda x: tuple(eval(x)))
    df['ISO'] = df['ISO'].fillna('NA')
    df.columns = ['fullName', 'BvD', 'ISO','order'] + list(df.columns[4:])
    if unweighted:
        df.drop_duplicates('BvD',inplace = True)
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
    df_hyper = pd.DataFrame(index=sorted(df.fullName.unique()))

    # Group the DataFrame by full name and get the unique sorted set of BvD values for each group
    hyper_companies = df.groupby('fullName')['BvD'].apply(lambda x: tuple(sorted(set(x))))

    # Group the DataFrame by full name and get the unique sorted set of ISO values for each group
    hyper_countries = df.groupby('fullName')['ISO'].apply(lambda x: tuple(sorted((x))))

    # Assign the hyper_companies and hyper_countries values to the corresponding columns in df_hyper
    df_hyper['BvD'] = hyper_companies
    df_hyper['ISO'] = hyper_countries 

    # Calculate the order of each hypergraph by counting the number of BvD values in each group
    df_hyper['order'] = hyper_companies.apply(lambda x: len(x))
    
    
    # Return the resulting hypergraph DataFrame
    return df_hyper


def remove_hubs(df,max_degree):
    df_bi =  get_bipartite_hyper(df)
    board_size = df_bi.groupby('BvD').size()
    busy_board = set(board_size[board_size>max_degree].index)
    df_bi = df_bi[(df['BvD'].isin(busy_board) == False)]
    return from_bipartite_to_hyper_df(df_bi)

