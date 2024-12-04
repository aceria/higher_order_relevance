from utilities import get_df_hyper,add_national_share
from itertools import combinations
from scipy.spatial import distance
import numpy as np
import pandas as pd, seaborn as sns
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import matplotlib.pyplot as plt
import networkx as nx
from randomization import randomize_hyper_df
import joblib

def get_joint_distr_dict(df, max_order=50):
    """
    Calculate the joint distribution dictionary for a given dataframe.

    Parameters:
    - df: pandas DataFrame
        The input dataframe containing the data.
    - max_order: int, optional
        The maximum order to consider. Default is 50.

    Returns:
    - joint_distr_dict: dict
        A dictionary containing the joint distribution for each 'links' value.
        The keys are the 'links' values and the values are numpy arrays representing the joint distribution.
    """
    joint_distr_dict = {}
    for iso, df_rest in df[df['order'] <= max_order].explode('links').groupby('links'):
        df_rest1 = df_rest.copy()
        print(iso)  # Print the current 'links' value
        df_rest1['national_count'] = df_rest1['national_count'].apply(lambda x: x[iso])  # Extract the count for the current 'links' value
        joint_prob = np.zeros((max_order + 1, max_order + 1))  # Initialize the joint probability matrix
        joint_prob_df = df_rest1.groupby(['order', 'national_count']).size()  # Group by 'order' and 'national_count' and calculate the size
        for x, y in joint_prob_df.index:
            joint_prob[x, y] = joint_prob_df.loc[(x, y)]  # Assign the size to the corresponding position in the joint probability matrix
        joint_distr_dict[iso] = joint_prob  # Store the joint probability matrix for the current 'links' value
    
    return joint_distr_dict



def compute_JS_from_joint_distr(joint_distr_dict):
    """
    Compute the Jensen-Shannon divergence between pairs of joint distributions.

    Parameters:
    joint_distr_dict (dict): A dictionary containing joint distributions.

    Returns:
    JS_df_overall (pandas.DataFrame): A DataFrame containing the Jensen-Shannon divergence values between pairs of joint distributions.
    """
    # Create an empty DataFrame with index and columns sorted
    JS_df_overall = pd.DataFrame(index=sorted(joint_distr_dict.keys()), columns=sorted(joint_distr_dict.keys()))
    
    # Compute the Jensen-Shannon divergence between pairs of joint distributions
    for iso1, iso2 in combinations(JS_df_overall.columns, 2):
        JS_df_overall.loc[iso1, iso2] = distance.jensenshannon(joint_distr_dict[iso1].flatten(), joint_distr_dict[iso2].flatten(), base=2.0)
        JS_df_overall.loc[iso2, iso1] = JS_df_overall.loc[iso1, iso2]  # Fill the symmetric position
        JS_df_overall.loc[iso1, iso1] = JS_df_overall.loc[iso2, iso2] = 0  # Set the diagonal elements to 1
        
    return JS_df_overall.astype(float)  # Convert the DataFrame to float type


def plot_heatmap(joint_distr_dict, JS_df_overall, title_str='', top_n=50, save=False):
    """
    Plot a heatmap of the joint distribution.

    Parameters:
    joint_distr_dict (dict): A dictionary containing the joint distribution data.
    JS_df_overall (DataFrame): A DataFrame containing the joint similarity values.
    title_str (str, optional): Title for the plot. Defaults to an empty string.
    top_n (int, optional): Number of top countries to include in the plot. Defaults to 50.
    save (bool, optional): Whether to save the plot as a PDF file. Defaults to False.

    Returns:
    None
    """
    # Select the top_n countries based on the joint distribution
    if top_n is not None:
        top_n_iso = pd.Series({k:joint_distr_dict[k].sum() for k in joint_distr_dict.keys()}).sort_values(ascending = False).index[:top_n]
    else:
        top_n_iso = np.array(joint_distr_dict.keys())
    
    # Filter the JS_df_overall DataFrame based on the top_n countries
    JS_df_overall = JS_df_overall.loc[top_n_iso, top_n_iso]
    
    # Compute the linkage matrix for hierarchical clustering
    linkage = hc.linkage(sp.distance.squareform(JS_df_overall), method='average')
    
    # Plot the heatmap using seaborn clustermap
    sns.clustermap(JS_df_overall, row_linkage=linkage, col_linkage=linkage, figsize=(15, 15))
    
    # Save the plot as a PDF file if save is True
    if save:
        plt.savefig('/data/ceriaac/CORPNET_project/Plots/heatmap_JS_joint_top_' + str(top_n) + '_' + title_str + '+.pdf', format='pdf')
    
    # Show the plot
    plt.show()


### general routine for running the analysis on the joint distribution of appointments

def run_analysis_joint_distr(size_lim=50, revenue_limit='overall', type_data='board', max_order=50, top_n=50, save_obj=False, save=False, randomization=False, seed1=0,unweighted = True):
    """
    Run analysis on joint distribution.

    Parameters:
    - size_lim (int): Size limit for the data.
    - revenue_limit (str): Revenue limit for the data.
    - type_data (str): Type of data to analyze: 'board', 'collaboration_diff_iso', or 'collaboration_max_iso'.
    - max_order (int): Maximum order for the joint distribution.
    - top_n (int): Number of top elements to consider.
    - save_obj (bool): Whether to save the joint distribution dictionary as a joblib file.
    - save (bool): Whether to save the heatmap plot.
    - randomization (bool): Whether to perform randomization on the data.
    - seed1 (int): Seed value for randomization.

    Returns:
    - joint_distr_dict_overall (dict): Joint distribution dictionary.
    """
    if type_data == 'board':
        type_data_str = 'inconsistent_ISO'
    else:
        type_data_str = type_data

    # Get the dataframe with specified parameters
    df = get_df_hyper(size_lim=size_lim, revenue_limit=revenue_limit, type_data=type_data,unweighted = unweighted)
    rand_str = ''
    
    # Perform randomization if specified
    if randomization:
        df = randomize_hyper_df(df, seed1=seed1, method=randomization)
        rand_str = '_rand_' + str(seed1)
    
    # Add national share to the dataframe
    df = add_national_share(df)
    
    # Calculate the joint distribution dictionary
    joint_distr_dict_overall = get_joint_distr_dict(df, max_order=max_order)
    
    # Compute the Jensen-Shannon divergence between pairs of joint distributions
    JS_df_overall = compute_JS_from_joint_distr(joint_distr_dict_overall)
    
    # Plot the heatmap of the joint distribution
    plot_heatmap(joint_distr_dict_overall, JS_df_overall, title_str=revenue_limit+'_'+type_data_str, top_n=top_n, save=save)
    if unweighted:
        type_data = type_data + '_unw'
    # Save the joint distribution dictionary as a joblib file if specified
    if save_obj:
        joblib.dump(joint_distr_dict_overall, '/data/ceriaac/CORPNET_project/Data/'+type_data_str+'/joint_distr/joint_distr' + str(size_lim) + '_' + revenue_limit + rand_str + type_data + '.joblib')
    
    # Return the joint distribution dictionary
    return joint_distr_dict_overall
#### couples of companies with same interlocking directors

def compute_order_set(df_hyper, order_link=2, size_lim=50, type_analysis='overall'):
    """
    Compute the order set for a given dataframe.

    Args:
        df_hyper (DataFrame): The input dataframe.
        order_link (int, optional): The order of the link. Defaults to 2.
        max_order (int, optional): The maximum order. Defaults to 50.
        type_analysis (str, optional): The type of analysis. Defaults to 'overall'.

    Returns:
        dict or tuple: The computed order set based on the type of analysis.
    """
    # Filter the dataframe based on the maximum order
    hyperlink_set1 = df_hyper[(df_hyper['order'] > 1) & (df_hyper['order'] <= size_lim)].drop_duplicates('BvD').copy()

    # Create a new column 'BvD_couples' which contains combinations of 'BvD' values
    hyperlink_set1['BvD_couples'] = hyperlink_set1['BvD'].apply(lambda x: list(combinations(x, order_link)))

    if type_analysis == 'country_level':
        # Create a new column 'ISO_couples' which contains combinations of 'ISO' values
        hyperlink_set1['ISO_couples'] = hyperlink_set1['ISO'].apply(lambda x: list(combinations(x, order_link)))

        

        list_explode = ['ISO_couples','BvD_couples']

    else:
        list_explode = ['BvD_couples']
    print(hyperlink_set1.columns)
    # Explode the 'BvD_couples' column to create multiple rows for each combination
    hyperlink_set1 = hyperlink_set1.explode(list_explode)

# Check if the 'BvD_couples' values are sorted
    assert hyperlink_set1['BvD_couples'].apply(lambda x: x == tuple(sorted(x))).all()
     

    if type_analysis == 'country_level':

        # Check if the 'ISO_couples' values are sorted
        assert hyperlink_set1['ISO_couples'].apply(lambda x: x == tuple(sorted(x))).all()

        # Filter the dataframe to get national and international hyperlink sets
        hyperlink_set_national = hyperlink_set1[hyperlink_set1['ISO_couples'].apply(lambda x: all(y == x[0] for y in x))]
        hyperlink_set_international = hyperlink_set1[hyperlink_set1['ISO_couples'].apply(lambda x: any(y != x[0] for y in x))]

        dict_national_order_set = dict()
        dict_international_order_set = dict()

        # Group the national hyperlink set by 'ISO_couples' and compute the order set for each order
        for iso, df_iso in hyperlink_set_national.explode('ISO_couples').groupby('ISO_couples'):
            dict_national_order_set[iso] = df_iso.groupby('order')['BvD_couples'].apply(lambda x: set(list(x))).to_dict()

        # Group the international hyperlink set by 'ISO_couples' and compute the order set for each order
        for iso, df_iso in hyperlink_set_international.explode('ISO_couples').groupby('ISO_couples'):
            dict_international_order_set[iso] = df_iso.groupby('order')['BvD_couples'].apply(lambda x: set(list(x))).to_dict()

        return dict_national_order_set, dict_international_order_set

    elif type_analysis == 'overall':
        dict_order_set = hyperlink_set1.groupby('order')['BvD_couples'].apply(lambda x: set(list(x))).to_dict()

        # Group the dataframe by 'BvD couples' and compute the order set for each order

        return dict_order_set
    
    





## compute overlapp
def compute_overlap(dict_G,size_lim = 50):

    dict_overlap= {}
    for order1 in range(2,size_lim+1):
        if order1 not in dict_G.keys():
                dict_G[order1] = []
        for order2 in range(2,size_lim+1):
            if order2 not in dict_G.keys():
                dict_G[order2] = []
            #print ((order1,order2))
            if order2>order1:
                
                
                
                sr = len(set(dict_G[order1]) & set(dict_G[order2]))
                sr_max = len(set(dict_G[order1]) | set(dict_G[order2]))
                dict_overlap[order1,order2] = (sr,sr_max)
                dict_overlap[order2,order1] = (sr,sr_max)
                len_set1 = len(set(dict_G[order1]))
                len_set2 = len(set(dict_G[order2]))
                dict_overlap[order1,order1] = (len_set1,len_set1)
                dict_overlap[order2,order2] = (len_set2,len_set2) 
                
                

    return dict_overlap





def run_analysis_overlap(size_lim=50, revenue_limit='overall', type_data='board', save=True, order_link=2, randomization=False, seed1=0, type_analysis=['overall', 'country_level'], title_str='', unweighted = False):
    """
    Run analysis to compute overlap between order sets.

    Parameters:
    - size_lim (int): The size limit for the analysis.
    - revenue_limit (str): The revenue limit for the analysis.
    - type_data (str): The type of data to analyze: 'board', 'collaboration_diff_iso', or 'collaboration_max_iso'.
    - save (bool): Whether to save the results.
    - order_link (int): The order link for the analysis.
    - randomization (bool): Whether to perform randomization.
    - seed1 (int): The seed for randomization.
    - type_analysis (list): The type of analysis to perform.
    - title_str (str): The title string for the analysis.

    Returns:
    - results (dict): A dictionary containing the results of the analysis.
    """
    if type_data == 'board':
        type_data_str = 'inconsistent_ISO'
    else:
        type_data_str = type_data
    # Get the dataframe for analysis
    df = get_df_hyper(size_lim=size_lim, revenue_limit=revenue_limit, type_data = type_data,unweighted = unweighted)

    # Initialize variables
    rand_str = ''
    title_str = title_str + '_' + str(order_link)
    if unweighted:
        title_str = title_str + '_unw'
    # Perform randomization if specified
    if randomization:
        df = randomize_hyper_df(df, seed1=seed1, method=randomization)
        rand_str = '_rand_' + str(randomization) + '_' + 'seed' + str(seed1)

    # Add national share to the dataframe
    df = add_national_share(df)

    # Initialize results dictionary
    results = dict()

    # Perform analysis for each type of analysis specified
    for analysis in type_analysis:
        if analysis == 'country_level':
            # Perform analysis for country level overlap
            national_overlap = {}
            international_overlap = {}
            dict_national_order_set, dict_international_order_set = compute_order_set(df, size_lim=size_lim, order_link=order_link, type_analysis=analysis)

            # Compute overlap for each country in the national order set
            for iso in dict_national_order_set.keys():
                national_overlap[iso] = compute_overlap(dict_national_order_set[iso], size_lim=size_lim)
            print('national analysis complete')

            # Compute overlap for each country in the international order set
            for iso in dict_international_order_set.keys():
                international_overlap[iso] = compute_overlap(dict_international_order_set[iso], size_lim=size_lim)
            print('international analysis complete')

            # Store the results in the results dictionary
            results['national_overlap'] = national_overlap
            results['international_overlap'] = international_overlap

            # Save the results if specified
            if save:
                joblib.dump(national_overlap, '/data/ceriaac/CORPNET_project/Data/'+type_data_str+'/overlapp/national_overlap_' + str(size_lim) + '_' + revenue_limit + title_str + type_data + '.joblib')
                joblib.dump(international_overlap, '/data/ceriaac/CORPNET_project/Data/'+type_data_str+'/overlapp/international_overlap_' + str(size_lim) + '_' + revenue_limit + title_str + type_data + '.joblib')

        if analysis == 'overall':
            # Perform analysis for overall overlap
            dict_order_set = compute_order_set(df, size_lim=size_lim, order_link=order_link, type_analysis=analysis)
            dict_overlap = compute_overlap(dict_order_set, size_lim=size_lim)
            print('overall analysis complete')

            # Store the results in the results dictionary
            results['overall_overlap'] = dict_overlap

            # Save the results if specified
            if save:
                joblib.dump(dict_overlap, '/data/ceriaac/CORPNET_project/Data/'+type_data_str+'/overlapp/overall_overlap_' + str(size_lim) + '_' + revenue_limit + title_str + rand_str + type_data + '.joblib')

    return results
  

