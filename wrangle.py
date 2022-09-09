import os
import env


import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sympy import *
# import pydataset

import seaborn as sns
from sympy.matrices import Matrix
from IPython.display import display

from functools import reduce
from itertools import combinations , product
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
import sklearn.preprocessing
from sklearn.model_selection import train_test_split




def sql_database_info_probe(schema_input='zillow'):
    '''
    just a way of exploring all the tables within a schema to save time. It also gives the size in MB


    '''

    schema = schema_input

    query_1 = f'''
    SELECT table_schema "Database Name",
    ROUND(SUM(data_length + index_length) / 1024 / 1024, 4) "DB Size in (MB)" 
    FROM information_schema.tables
    WHERE table_schema= "{schema}" 
    GROUP BY table_schema
    ;
    '''

    query_2 = f'''
    SELECT table_name AS "Tables",
    ROUND(((data_length + index_length) / 1024 / 1024), 4) AS "Size (MB)"
    FROM information_schema.TABLES
    WHERE table_schema = "{schema}"
    ORDER BY (data_length + index_length) DESC;
    '''

    info1 = pd.read_sql(query_1, get_db_url(schema))
    info2 = pd.read_sql(query_2, get_db_url(schema))

    display(f'In {schema} your overlall size(MB) is:', info1)
    tablenames = [x[0] for x in [list(i) for i in info2.values]]
    display(
        f'In {schema} you have the following table names and their sizes:', info2)
    x = []
    [x.append(pd.read_sql(f'describe {i}', get_db_url(schema)))for i in tablenames]
    [display(sympify(f'{(tablenames[i]).capitalize()}'), k)
     for i, k in enumerate(x)]
    



def remove_outliers_v2(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe. This was a gift from our instructor. It is an implementation of the Tukey method.
        https://en.wikipedia.org/wiki/Tukey%27s_range_test
    '''
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df['outlier'] = False
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    df = df[df.outlier == False]
  
    print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df

def get_db_url(db, env_file=env):
    '''
    returns a formatted string ready to utilize as a sql url connection

    args: db: a string literal representing a schema
    env_file: bool: checks to see if there is an env.py present in the cwd

    make sure that if you have an env file that you import it outside of the scope 
    of this function call, otherwise env.user wont mean anything ;)
    '''
    if env_file:
        username, password, host = (env.username, env.password, env.host)
        return f'mysql+pymysql://{username}:{password}@{host}/{db}'
    else:
        return 'you need some credentials to access a database usually and I dont want you to type them here.'

def new_zillow_2017():
    '''
    Here I selected every colomun I could where the non-null was greater than 50k to minimize the data cleaning,
    additionally I did not need to join the third table as I only needed the propertylandusetypeid type from them. I was able to
    obtain that through an exploritory call.


    '''

    schema='zillow'

    query='''
    select
    assessmentyear,
    bathroomcnt,
    bedroomcnt,
    calculatedbathnbr,
    calculatedfinishedsquarefeet,
    fips,
    fullbathcnt,
    prop.id,
    latitude,
    logerror,
    longitude,
    lotsizesquarefeet,
    taxvaluedollarcnt,
    transactiondate,
    yearbuilt
    from
    properties_2017 prop
    join
    predictions_2017 pred
    on 
    prop.id = pred.id
    where
    (propertylandusetypeid=261	
    or
    propertylandusetypeid=279
    and
    transactiondate like '2017')
    
    
    
    '''

    
    
    
    
    return pd.read_sql(query, get_db_url(schema))










def get_zillow_2017():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow_2017.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow_2017.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_2017()
        
        # Cache data
        df.to_csv('zillow_2017.csv')
        
    return df








def decademap(x):
    'makes a decade col when combined with an apply function '
    yearsgrouped=np.arange(1800,2020,10)
    if x >= yearsgrouped[21]:
        decade=yearsgrouped[21]
        return decade
    elif x >= yearsgrouped[20]:
        decade=yearsgrouped[20]
        return decade
    elif x >= yearsgrouped[19]:
        decade=yearsgrouped[19]
        return decade
    elif x >= yearsgrouped[18]:
        decade=yearsgrouped[18]
        return decade
    elif x >= yearsgrouped[17]:
        decade=yearsgrouped[17]
        return decade
    elif x >= yearsgrouped[16]:
        decade=yearsgrouped[16]
        return decade
    elif x >= yearsgrouped[15]:
        decade=yearsgrouped[15]
        return decade
    elif x >= yearsgrouped[14]:
        decade=yearsgrouped[14]
        return decade
    elif x >= yearsgrouped[13]:
        decade=yearsgrouped[13]
        return decade
    elif x >= yearsgrouped[12]:
        decade=yearsgrouped[12]
        return decade
    elif x >= yearsgrouped[11]:
        decade=yearsgrouped[11]
        return decade
    elif x >= yearsgrouped[10]:
        decade=yearsgrouped[10]
        return decade
    elif x >= yearsgrouped[9]:
        decade=yearsgrouped[9]
        return decade
    elif x >= yearsgrouped[8]:
        decade=yearsgrouped[8]
        return decade
    elif x >= yearsgrouped[7]:
        decade=yearsgrouped[7]
        return decade
    elif x >= yearsgrouped[6]:
        decade=yearsgrouped[6]
        return decade
    elif x >= yearsgrouped[5]:
        decade=yearsgrouped[5]
        return decade
    elif x >= yearsgrouped[4]:
        decade=yearsgrouped[4]
        return decade
    elif x >= yearsgrouped[3]:
        decade=yearsgrouped[3]
        return decade
    elif x >= yearsgrouped[2]:
        decade=yearsgrouped[2]
        return decade
    elif x >= yearsgrouped[1]:
        decade=yearsgrouped[1]
        return decade
    elif x >= yearsgrouped[0]:
        decade=yearsgrouped[0]
        return decade





def train_validate_test(df, target,sizearr=[.2,.3],random_state=123):
    """
    this function takes in a dataframe and splits it into 3 samples,
    a test, which is 20% of the entire dataframe,
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe.
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    """
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=sizearr[0], random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=sizearr[1], random_state=123)

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]

    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]

    return X_train, y_train, X_validate, y_validate, X_test, y_test


def calfipsmapper(df):
    calfipsdf=pd.read_csv('califips.csv')
    a=calfipsdf.columns[0]
    b=calfipsdf.columns[1]
    calfipsdict=dict(zip(calfipsdf[a],calfipsdf[b]))
    df['county']= df["fips"].map(calfipsdict)
    df.county=df.county.map({' Los Angeles County':'LA',' Orange County':'Orange'})
    df.rename(columns={'calculatedfinishedsquarefeet':'area'},inplace=True)
    return df



 