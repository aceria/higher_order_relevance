from utilities import get_df_hyper,add_same_label_share
from itertools import combinations
import numpy as np
import pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
from randomization import randomize_hyper_df
import joblib

def compute_joint_metrics(m1):
    numerator = ((np.indices(m1.shape)[0][2:,1:] -np.indices(m1.shape)[1][2:,1:]) * m1[2:,1:]).sum()
    denominator = (((np.indices(m1.shape)[0][2:,1:] - np.indices(m1.shape)[1][2:,1:])[:,0]) *m1[2:,1:].sum(axis = 1)).sum()
    return numerator/denominator

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
    for labels, df_rest in df[df['order'] <= max_order].explode('links').groupby('links'):
        df_rest1 = df_rest.copy()
        print(labels)  # Print the current 'links' value
        df_rest1['same_label_count'] = df_rest1['same_label_count'].apply(lambda x: x[labels])  # Extract the count for the current 'links' value
        joint_prob = np.zeros((max_order + 1, max_order + 1))  # Initialize the joint probability matrix
        joint_prob_df = df_rest1.groupby(['order', 'same_label_count']).size()  # Group by 'order' and 'same_label_count' and calculate the size
        for x, y in joint_prob_df.index:
            joint_prob[x, y] = joint_prob_df.loc[(x, y)]  # Assign the size to the corresponding position in the joint probability matrix
        joint_distr_dict[labels] = joint_prob  # Store the joint probability matrix for the current 'links' value
    
    return joint_distr_dict









### general routine for running the analysis on the joint distribution of appointments

def run_analysis_joint_distr(link_list,label_list, max_order=50, top_n=False, save_obj=False, randomization=False, seed1=0,unweighted = True, show_heatmap = False):
    """
    Run analysis on joint distribution.

    Parameters:
    - size_lim (int): Size limit for the data.
    - max_order (int): Maximum order for the joint distribution.
    - top_n (int): Number of top elements to consider.
    - save_obj (bool): Whether to save the joint distribution dictionary as a joblib file.
    - save (bool): Whether to save the heatmap plot.
    - randomization (bool): Whether to perform randomization on the data.
    - seed1 (int): Seed value for randomization.

    Returns:
    - joint_distr_dict_overall (dict): Joint distribution dictionary.
    """
   

    # Get the dataframe with specified parameters
    df = get_df_hyper(link_list, label_list, size_lim=max_order, unweighted = unweighted)
    rand_str = ''
    
    # Perform randomization if specified
    if randomization:
        df = randomize_hyper_df(df, seed1=seed1, method=randomization)
        rand_str = '_rand_' + str(seed1)
    
    # Add same_label share to the dataframe
    df = add_same_label_share(df)
    
    # Calculate the joint distribution dictionary
    joint_distr_dict_overall = get_joint_distr_dict(df, max_order=max_order)
    group_balance = {k: compute_joint_metrics(joint_distr_dict_overall[k]) for k in joint_distr_dict_overall.keys()}
    
    
    unw_str =''
    if unweighted:
        unw_str = '_unw'
    
    joint_distr_df = {k: pd.DataFrame(joint_distr_dict_overall[k]/(joint_distr_dict_overall[k].sum())).replace(0, np.nan).loc[2:max_order,1:max_order] for k in joint_distr_dict_overall.keys()}
    if show_heatmap:
        for k in joint_distr_dict_overall:
            data = joint_distr_df[k]
            df_joint = pd.DataFrame(joint_distr_dict_overall[k])
            mean1 = pd.concat([df_joint[k]*k for k in df_joint],axis = 1).apply(sum,axis = 1)/df_joint.apply(sum,axis = 1)
            ax_heatmap3 = sns.heatmap(np.log10(data), cmap='vlag',cbar_kws={'label': r'$Log(P_{\ell}(d,k))$'})
            # Plot the mean line
            ax_heatmap3.plot(mean1.loc[2:].values -0.5, mean1.loc[2:].index -1.5, color='k', label= r'$\langle k_{\ell} \rangle_d$',linewidth=0.7)
            ax_heatmap3.legend(loc='upper right', bbox_to_anchor=(0.7, 1.05))
            ax_heatmap3.set_xlabel('same label nodes $k$')
            ax_heatmap3.set_ylabel('order $d$')
            plt.title(f'Joint distribution of appointments for {k}')
            plt.show()
    # Save the joint distribution dictionary as a joblib file if specified
    if save_obj:
        joblib.dump(joint_distr_dict_overall, 'Data/joint_distr' + str(max_order) + '_' + rand_str +unw_str+'.joblib')
    
    # Return the joint distribution dictionary
    return {'group_composition_distribution':joint_distr_df,'group_balance':group_balance}
#### couples of companies with same interlocking directors

   