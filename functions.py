import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display

def clean_data(df, y_label=None):
    '''
    INPUT
    df - pandas dataframe 
    
    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector
    
    Perform to obtain the correct X and y objects
    This function cleans df using the following steps to produce X and y:
    1. Drop all the rows with no salaries
    2. Create X as all the columns that are not the Salary column
    3. Create y as the Salary column
    4. Drop the Salary, Respondent, and the ExpectedSalary columns from X
    5. For each numeric variable in X, fill the column with the mean value of the column.
    6. Create dummy columns for all the categorical variables in X, drop the original columns
    '''
    
    
    # 3. Create y as the Salary column
    if y_label:
        y = df[y_label]

        # 4. Drop the Salary, Respondent, and the ExpectedSalary columns from X
        df = df.drop([y_label], axis=1)
    else:
        y = None
    
    # 6. Create dummy columns for all the categorical variables in X, drop the original columns
    cat_vars = df.select_dtypes(include=['object']).copy().columns
    for var in  cat_vars:
        # for each cat add dummy var, drop original column
        df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var, prefix_sep='_', drop_first=False, dummy_na=True)], axis=1)

    X = df
    return X, y




def add_iqr(describe):
    '''
    INPUT:
    df_describe - dataframe from df.describe()
    OUTPUT:
    new_df_describe - a dataframe with added IQR to index
    calculates the IQR with 75% - 25% quantile
    '''
    describe.loc['IQR'] = describe.loc['75%'] - describe.loc['25%']
    
    return describe



def remove_outlier(df, columns):
    '''
    INPUT:
    df - the pandas dataframe you want to remove outliers
    columns - the columns you want to remove outliers, dtype=list
    OUTPUT:
    new_df - a dataframe with removed outliers for specified columns
    '''
    describe = add_iqr(df.describe())
    index = df.index
    for col in columns:
        iqr = describe.loc['IQR', col]
        lower = describe.loc['25%', col]
        upper = describe.loc['75%', col]
        new_index = df[df[col].between(lower - 1.5*iqr, upper + 1.5*iqr)].index
        index=np.intersect1d(index, new_index)
        
    return df[df.index.isin(index)]


    
# Calculate the explained variance for the top n principal components
# you may assume you have access to the global var N_COMPONENTS
def explained_variance(s, n_top_components):
    '''Calculates the approx. data variance that n_top_components captures.
       :param s: A dataframe of singular values for top components; 
           the top value is in the last row.
       :param n_top_components: An integer, the number of top components to use.
       :return: The expected data variance covered by the n_top_components.'''
    
    # calculate approx variance
    exp_variance = np.square(s.iloc[:n_top_components,:]).sum()/np.square(s).sum()
    
    return exp_variance[0]


def display_components(v, features_list, component_num, n_weights=5):
    
    # get index of component (last row - component_num)
    row_idx = component_num
    
    # get the list of weights from a row in v
    v_1_row = v.iloc[row_idx]
    v_1 = np.squeeze(v_1_row.values)
    # match weights to features in baseline dataframe, using list comprehension
    comps = pd.DataFrame(list(zip(v_1, features_list)),
                         columns=['weights', 'features'])
    
    # we will want to sort by the weights
    # weight can be neg/pos and we sort by magnitude
    comps['abs_weight'] = comps['weights'].apply(lambda x: np.abs(x))
    sorted_weight_data = comps.sort_values(by='abs_weight', ascending=False).head(n_weights)
    sns.barplot(data=sorted_weight_data, 
                x='weights', 
                y='features', 
                palette='Blues_d')



