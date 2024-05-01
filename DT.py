import pandas as pd
import numpy as np

data = pd.read_csv("test.csv")
print(data.head())

def gini_impurity(y):
    '''
    Given a Pandas Series, it calculates the Gini Impurity. 
    y: variable with which calculate Gini Impurity.
    '''
    if isinstance(y, pd.Series):
        p = y.value_counts()/y.shape[0]
        gini = 1-np.sum(p**2)
        return(gini)

    else:
        raise('Object must be a Pandas Series.')

#print(gini_impurity(data.buy) )

def entropy(y):
    '''
    Given a Pandas Series, it calculates the entropy. 
    y: variable with which calculate entropy.
    '''
    if isinstance(y, pd.Series):
        a = y.value_counts()/y.shape[0]
        entropy = np.sum(-a*np.log2(a+1e-9))
        return(entropy)

    else:
        raise('Object must be a Pandas Series.')

#print(entropy(data.buy))
