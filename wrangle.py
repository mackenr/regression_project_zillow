import datetime as dt
import os
import time
from functools import reduce
from itertools import combinations, product

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from IPython.display import display
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sympy import *
from sympy.matrices import Matrix

import env

# import pydataset


# prepped=prep_zillow_2017()


def prep_zillow_2017(k=1.25):
    """
    returns df



    """

    df = get_zillow_2017()
    fullsfips = df.fips.value_counts()
    print(f"our sql grab:\n {df.shape}")

    # lista=df[df.isnull()==False].count().nlargest(n=23,keep='all').index.tolist()
    # lista.sort()
    df.transactiondate = pd.to_datetime(df.transactiondate, yearfirst=True)
    df["transaction_month"] = df.transactiondate.dt.month_name()
    df["transaction_month_int"] = df.transactiondate.apply(monthmap)

    df = df[df.transactiondate < "2018-01-01"]
    df = df[df.transactiondate >= "2017-01-01"]

    print(f"after ensuring dates sql grab:\n {df.shape}")
    ## ensures the dates of transaction are where they need to be
    print(
        f"df.transactiondate.min():\n{df.transactiondate.min()}\n\ndf.transactiondate.max():\n{df.transactiondate.max()}"
    )
    df["transactiondate_in_days"] = (
        df.transactiondate - dt.datetime(2017, 1, 1)
    ).dt.days
    df["transactiondate_in_days"] = (
        df["transactiondate_in_days"] + 1
    )  # had issues will zeros

    dfj = df

    mwp = [
        "bathroomcnt",
        "bedroomcnt",
        "calculatedfinishedsquarefeet",
        "taxvaluedollarcnt",
        "fips",
        "yearbuilt",
        "parcelid",
        "lotsizesquarefeet",
    ]
    drop = [
        "bathroomcnt",
        "bedroomcnt",
        "calculatedfinishedsquarefeet",
        "taxvaluedollarcnt",
        "fips",
        "yearbuilt",
        "lotsizesquarefeet",
    ]

    todrop = set(drop)
    # mwp=['bathroomcnt','bedroomcnt','calculatedfinishedsquarefeet','taxvaluedollarcnt','fips','yearbuilt']
    colsdfj = set(dfj.columns)
    dfjkeep = list(colsdfj - todrop)

    dfj = dfj[dfjkeep]

    df = df[mwp]
    df_6037 = df[df.fips == 6037]
    df_6059 = df[df.fips == 6059]
    df_6111 = df[df.fips == 6111]

    x1 = len(df)

    cols = df.columns.to_list()
    # Here I seperated by each fips(county) removed the outliesrs from each then remerged them using an outer joint to protect the keys of each. This was done to protect the indvidual integrity of each subgroup

    df_6037 = remove_outliers_v2(df=df_6037, k=k, col_list=cols)
    df_6037.drop(columns=["outlier"], inplace=True)

    df_6059 = remove_outliers_v2(df=df_6059, k=k, col_list=cols)
    df_6059.drop(columns=["outlier"], inplace=True)

    df_6111 = remove_outliers_v2(df=df_6111, k=k, col_list=cols)
    df_6111.drop(columns=["outlier"], inplace=True)

    df = pd.merge(df_6037, df_6059, how="outer", on=mwp)
    df = pd.merge(df, df_6111, how="outer", on=mwp)

    # here I did just made some changes for readbility
    df.dropna(inplace=True)

    df.fips = df.fips.astype(int)
    df["decade"] = df.yearbuilt.apply(decademap)
    ## Actual Percent Change

    meankurt = df.kurt().mean()

    # Merge two DataFrames by index using pandas.merge()
    ## This merge just brings back all the additional non numeric cols using an inner merge so the nulls are dropped this was required because the tukey method as implempmented onlyh works with numeric data
    df = pd.merge(df, dfj, on="parcelid", how="inner")

    # display(df.head(),df.info(verbose=True),df.describe(include='all'),df.shape)
    x2 = len(df)
    percentchangeafterdrop = round(((x2 - x1) / x1) * 100, 2)

    # here I did some feature engineering to see if there was any observational changes in relationships

    df["lotsize/area"] = df.lotsizesquarefeet / df.calculatedfinishedsquarefeet
    # df['bedbath_harmean']=(((df.bedroomcnt)**-1+(df.bathroomcnt**-1))*2)
    # # df['bedbath_harmeandividesarea']=df.calculatedfinishedsquarefeet/ df['bedbath_harmean']
    # df['sqrt(bed^2+bath^2)']=((df.bedroomcnt**2+df.bathroomcnt**2)**(1/2))
    # df['sqrt(bed^2+bath^2)dividesarea']=df.calculatedfinishedsquarefeet/ df['sqrt(bed^2+bath^2)']
    # df['bathplusbathdividesarea']=df.calculatedfinishedsquarefeet/(df.bathroomcnt+df.bedroomcnt)
    # df['sqrt(bed^2+bath^2)divides(lotsize/area)']= df['lotsize/area']/df['sqrt(bed^2+bath^2)']
    # df['bedbath_harmean)divides(lotsize/area)']= df['lotsize/area']/ df['bedbath_harmean']
    df["latscaled"] = df.latitude * 1e-6
    df["longscaled"] = df.latitude * 1e-6
    df["latlongPythagC"] = (df["longscaled"] ** 2 + df["latscaled"] ** 2) ** 0.5

    df = calfipsmapper(df)
    lat_long_range = list(np.arange(0, 1, 0.04))
    lat_long_range = df.latlongPythagC.quantile(
        q=lat_long_range, interpolation="linear"
    ).tolist()

    df["Geogroups"] = df.latlongPythagC.map(lambda x: latCgroups(x, lat_long_range))
    le = LabelEncoder()
    df["Geogroups"] = le.fit_transform(df["Geogroups"])
    df["age"] = 2017 - df.yearbuilt
    df["agebydecade"] = 2017 - df.decade
    # df.drop(columns=['transactiondate','latlongPythagC','decade','assessmentyear','longscaled','latscaled','transaction_month_int','transactiondate_in_days','transactiondate_in_days'],inplace=True)

    df.rename(columns={"calculatedfinishedsquarefeet": "area"}, inplace=True)

    display(
        print(
            f"This is our percent change after removing all the outliers and merging :\n {percentchangeafterdrop}%\nmean kurt:\n{meankurt}\nfinal shape:\n{ df.shape}"
        ),
        (df[["parcelid", "county"]].groupby(by=["county"]).nunique() / len(df))
        .style.format(lambda x: f"{x*100:.2f}%")
        .set_caption(caption="Prepped Data \n Percentage per county"),
        fullsfips,
    )
    df.drop(
        columns=[
            "parcelid",
            "assessmentyear",
            "logerror",
            "calculatedbathnbr",
            "fullbathcnt",
        ],
        inplace=True,
    )

    # display(pd.DataFrame(df))
    cols = list(df.columns)
    # ensures the target variable is in the first position
    cols.remove("taxvaluedollarcnt")
    cols.insert(0, "taxvaluedollarcnt")
    df = df[cols]

    return df

    # return df


def sql_database_info_probe(schema_input="zillow"):
    """
    just a way of exploring all the tables within a schema to save time. It also gives the size in MB


    """

    schema = schema_input

    query_1 = f"""
    SELECT table_schema "Database Name",
    ROUND(SUM(data_length + index_length) / 1024 / 1024, 4) "DB Size in (MB)" 
    FROM information_schema.tables
    WHERE table_schema= "{schema}" 
    GROUP BY table_schema
    ;
    """

    query_2 = f"""
    SELECT table_name AS "Tables",
    ROUND(((data_length + index_length) / 1024 / 1024), 4) AS "Size (MB)"
    FROM information_schema.TABLES
    WHERE table_schema = "{schema}"
    ORDER BY (data_length + index_length) DESC;
    """

    info1 = pd.read_sql(query_1, get_db_url(schema))
    info2 = pd.read_sql(query_2, get_db_url(schema))

    display(f"In {schema} your overlall size(MB) is:", info1)
    tablenames = [x[0] for x in [list(i) for i in info2.values]]
    display(f"In {schema} you have the following table names and their sizes:", info2)
    x = []
    [x.append(pd.read_sql(f"describe {i}", get_db_url(schema))) for i in tablenames]
    [display(sympify(f"{(tablenames[i]).capitalize()}"), k) for i, k in enumerate(x)]


def remove_outliers_v2(df, k, col_list):
    """remove outliers from a list of columns in a dataframe
    and return that dataframe. This was a gift from our instructor. It is an implementation of the Tukey method.
    https://en.wikipedia.org/wiki/Tukey%27s_range_test
    """
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df["outlier"] = False
    for col in col_list:

        q1, q3 = df[col].quantile([0.25, 0.75])  # get quartiles

        iqr = q3 - q1  # calculate interquartile range

        upper_bound = q3 + k * iqr  # get upper bound
        lower_bound = q1 - k * iqr  # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df["outlier"] = np.where(
            ((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False),
            True,
            df.outlier,
        )

    df = df[df.outlier == False]

    print(f"Number of observations removed: {num_obs - df.shape[0]}")

    return df


def get_db_url(db, env_file=env):
    """
    returns a formatted string ready to utilize as a sql url connection

    args: db: a string literal representing a schema
    env_file: bool: checks to see if there is an env.py present in the cwd

    make sure that if you have an env file that you import it outside of the scope
    of this function call, otherwise env.user wont mean anything ;)
    """
    if env_file:
        username, password, host = (env.username, env.password, env.host)
        return f"mysql+pymysql://{username}:{password}@{host}/{db}"
    else:
        return "you need some credentials to access a database usually and I dont want you to type them here."


def new_zillow_2017():
    """
    Here I selected every colomun I could where the non-null was greater than 50k to minimize the data cleaning,
    additionally I did not need to join the third table as I only needed the propertylandusetypeid type from them. I was able to
    obtain that through an exploritory call.


    """

    schema = "zillow"

    query = """
    select
    assessmentyear,
    bathroomcnt,
    bedroomcnt,
    calculatedbathnbr,
    calculatedfinishedsquarefeet,
    fips,
    fullbathcnt,
    prop.parcelid,
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
    prop.parcelid = pred.parcelid
    where
    (propertylandusetypeid=261	
    or
    propertylandusetypeid=279
    and
    transactiondate like '2017')
    
    
    
    """

    # parcelid was the proper key to join on be careful

    return pd.read_sql(query, get_db_url(schema))


def get_zillow_2017():
    """
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    """
    if os.path.isfile("zillow_2017.csv"):

        # If csv file exists read in data from csv file.
        df = pd.read_csv("zillow_2017.csv", index_col=0)

    else:

        # Read fresh data from db into a DataFrame
        df = new_zillow_2017()

        # Cache data
        df.to_csv("zillow_2017.csv")

    return df


def decademap(x):
    "makes a decade col when combined with an apply function"
    yearsgrouped = np.arange(1800, 2020, 10)
    if x >= yearsgrouped[21]:
        decade = yearsgrouped[21]
        return decade
    elif x >= yearsgrouped[20]:
        decade = yearsgrouped[20]
        return decade
    elif x >= yearsgrouped[19]:
        decade = yearsgrouped[19]
        return decade
    elif x >= yearsgrouped[18]:
        decade = yearsgrouped[18]
        return decade
    elif x >= yearsgrouped[17]:
        decade = yearsgrouped[17]
        return decade
    elif x >= yearsgrouped[16]:
        decade = yearsgrouped[16]
        return decade
    elif x >= yearsgrouped[15]:
        decade = yearsgrouped[15]
        return decade
    elif x >= yearsgrouped[14]:
        decade = yearsgrouped[14]
        return decade
    elif x >= yearsgrouped[13]:
        decade = yearsgrouped[13]
        return decade
    elif x >= yearsgrouped[12]:
        decade = yearsgrouped[12]
        return decade
    elif x >= yearsgrouped[11]:
        decade = yearsgrouped[11]
        return decade
    elif x >= yearsgrouped[10]:
        decade = yearsgrouped[10]
        return decade
    elif x >= yearsgrouped[9]:
        decade = yearsgrouped[9]
        return decade
    elif x >= yearsgrouped[8]:
        decade = yearsgrouped[8]
        return decade
    elif x >= yearsgrouped[7]:
        decade = yearsgrouped[7]
        return decade
    elif x >= yearsgrouped[6]:
        decade = yearsgrouped[6]
        return decade
    elif x >= yearsgrouped[5]:
        decade = yearsgrouped[5]
        return decade
    elif x >= yearsgrouped[4]:
        decade = yearsgrouped[4]
        return decade
    elif x >= yearsgrouped[3]:
        decade = yearsgrouped[3]
        return decade
    elif x >= yearsgrouped[2]:
        decade = yearsgrouped[2]
        return decade
    elif x >= yearsgrouped[1]:
        decade = yearsgrouped[1]
        return decade
    elif x >= yearsgrouped[0]:
        decade = yearsgrouped[0]
        return decade


def train_validate_test(df, sizearr=[0.2, 0.3], random_state=123):
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
    train_validate, test = train_test_split(
        df, test_size=sizearr[0], random_state=random_state
    )

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(
        train_validate, test_size=sizearr[1], random_state=random_state
    )

    return train, validate, test


def bigX_little_y(train, validate, test, target):
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
    calfipsdf = pd.read_csv("califips.csv")
    a = calfipsdf.columns[0]
    b = calfipsdf.columns[1]
    calfipsdict = dict(zip(calfipsdf[a], calfipsdf[b]))
    df["county"] = df["fips"].map(calfipsdict)
    df.county = df.county.map(
        {
            " Los Angeles County": "LA",
            " Orange County": "Orange",
            " Ventura County": "Ventura",
        }
    )

    return df


def monthmap(x):
    """
    Created to obtain a month from a datatime
    """

    return x.month


def latCgroups(x, lat_long_range):
    "makes a lat_long_quantile col when combined with an apply function"

    if x >= lat_long_range[24]:
        lat_long_quantile = lat_long_range[24]
        return lat_long_quantile
    elif x >= lat_long_range[23]:
        lat_long_quantile = lat_long_range[23]
        return lat_long_quantile
    elif x >= lat_long_range[22]:
        lat_long_quantile = lat_long_range[22]
        return lat_long_quantile
    elif x >= lat_long_range[21]:
        lat_long_quantile = lat_long_range[21]
        return lat_long_quantile
    elif x >= lat_long_range[20]:
        lat_long_quantile = lat_long_range[20]
        return lat_long_quantile
    elif x >= lat_long_range[19]:
        lat_long_quantile = lat_long_range[19]
        return lat_long_quantile
    elif x >= lat_long_range[18]:
        lat_long_quantile = lat_long_range[18]
        return lat_long_quantile
    elif x >= lat_long_range[17]:
        lat_long_quantile = lat_long_range[17]
        return lat_long_quantile
    elif x >= lat_long_range[16]:
        lat_long_quantile = lat_long_range[16]
        return lat_long_quantile
    elif x >= lat_long_range[15]:
        lat_long_quantile = lat_long_range[15]
        return lat_long_quantile
    elif x >= lat_long_range[14]:
        lat_long_quantile = lat_long_range[14]
        return lat_long_quantile
    elif x >= lat_long_range[13]:
        lat_long_quantile = lat_long_range[13]
        return lat_long_quantile
    elif x >= lat_long_range[12]:
        lat_long_quantile = lat_long_range[12]
        return lat_long_quantile
    elif x >= lat_long_range[11]:
        lat_long_quantile = lat_long_range[11]
        return lat_long_quantile
    elif x >= lat_long_range[10]:
        lat_long_quantile = lat_long_range[10]
        return lat_long_quantile
    elif x >= lat_long_range[9]:
        lat_long_quantile = lat_long_range[9]
        return lat_long_quantile
    elif x >= lat_long_range[8]:
        lat_long_quantile = lat_long_range[8]
        return lat_long_quantile
    elif x >= lat_long_range[7]:
        lat_long_quantile = lat_long_range[7]
        return lat_long_quantile
    elif x >= lat_long_range[6]:
        lat_long_quantile = lat_long_range[6]
        return lat_long_quantile
    elif x >= lat_long_range[5]:
        lat_long_quantile = lat_long_range[5]
        return lat_long_quantile
    elif x >= lat_long_range[4]:
        lat_long_quantile = lat_long_range[4]
        return lat_long_quantile
    elif x >= lat_long_range[3]:
        lat_long_quantile = lat_long_range[3]
        return lat_long_quantile
    elif x >= lat_long_range[2]:
        lat_long_quantile = lat_long_range[2]
        return lat_long_quantile
    elif x >= lat_long_range[1]:
        lat_long_quantile = lat_long_range[1]
        return lat_long_quantile
    elif x >= lat_long_range[0]:
        lat_long_quantile = lat_long_range[0]
        return lat_long_quantile
