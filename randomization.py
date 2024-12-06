import random
import pandas as pd
import numpy as np
import networkx as nx
from utilities import from_bipartite_to_hyper_df,get_bipartite_hyper




def randomize_hyper_df(hyper_df,seed1=0,method = 3,return_bipartite = False,verbose = False,interlocking_directors = True):
    bipartite_df = get_bipartite_hyper(hyper_df)
    del hyper_df
    bipartite_df.columns = ['fullName','BvD','ISO']+ list(bipartite_df.columns)[3:]
    random.seed(seed1)
    if method == 0:
        random.shuffle(bipartite_df['fullName'].values)
        df_list = [bipartite_df]
    elif method != 0:
        if method == 1: rand_list = ['order']
        if method == 2: rand_list = ['ISO']
        elif method == 3: rand_list = ['order','ISO']
        df_list = list()
        for iso,df_rest in bipartite_df.groupby(rand_list):
            
            if verbose:print(iso)
            random.shuffle(df_rest['fullName'].values)            
            df_list.append(df_rest)
    rand_df = pd.concat(df_list)
    n_doubles = rand_df.shape[0]
    rand_df.drop_duplicates(['fullName','BvD'],inplace = True)
    del bipartite_df
    print('removed '+str(n_doubles-rand_df.shape[0])+' doubles')
    print('removed '+str((n_doubles-rand_df.shape[0])/n_doubles)+'ratio doubles')

    if interlocking_directors:
            interlocking_director = rand_df.groupby('fullName')['BvD'].apply(lambda x: len(set(x)))
            interlocking_director = set(interlocking_director[interlocking_director>1].index)
            rand_df = rand_df[rand_df['fullName'].isin(interlocking_director)]
    print('randomization has been done with method '+str(method) +' and seed '+str(seed1))
    
    if return_bipartite:
        return rand_df.sort_values('fullName')
    else:
        return from_bipartite_to_hyper_df(rand_df)
