
"""Statistical methods used for analyzing the shared bike data
"""

import logging
import daiquiri

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import xgboost as xgb

from jitenshea import config

daiquiri.setup(logging.INFO)
logger = daiquiri.getLogger("stats")

SEED = 2018
np.random.seed(SEED)

CLUSTER_ACT_PATH_CSV ='jitenshea/data/cluster_activite.csv'


###################################
###         CLUSTER ACTIVITE
###################################

def preprocess_data_for_clustering(df):
    """Prepare data in order to apply a clustering algorithm

    Parameters
    ----------
    df : pandas.DataFrame
        Input data, *i.e.* city-related timeseries, supposed to have
    `station_id`, `ts` and `nb_bikes` columns

    Returns
    -------
    pandas.DataFrame
        Simpified version of `df`, ready to be used for clustering

    """
    # Filter unactive stations
    max_bikes = df.groupby("station_id")["nb_bikes"].max()
    unactive_stations = max_bikes[max_bikes==0].index.tolist()
    active_station_mask = np.logical_not(df['station_id'].isin(unactive_stations))
    df = df[active_station_mask]
    # Set timestamps as the DataFrame index and resample it with 5-minute periods
    df = (df.set_index("ts")
          .groupby("station_id")["nb_bikes"]
          .resample("5T")
          .mean()
          .bfill())
    df = df.unstack(0)
    # Drop week-end records
    df = df[df.index.weekday < 5]
    # Gather data regarding hour of the day
    df['hour'] = df.index.hour
    df = df.groupby("hour").mean()
    return df / df.max()

def compute_clusters(df, cluster_path_csv=None):
    """Compute station clusters based on bike availability time series

    Parameters
    ----------
    df : pandas.DataFrame
        Input data, *i.e.* city-related timeseries, supposed to have
    `station_id`, `ts` and `nb_bikes` columns

    cluster_path_csv : String :
        Path to export df_labels DataFrame

    Returns
    -------
    If cluster_path_csv is not None :
        Return noting (just export DataFrame in cluster_path_csv)
    Else :
        dict
            Two pandas.DataFrame, the former for station clusters and the latter
        for cluster centroids


    """
    df_norm = preprocess_data_for_clustering(df)
    model = KMeans(n_clusters=4, random_state=SEED)
    kmeans = model.fit(df_norm.T)
    df_labels = pd.DataFrame({"id_station": df_norm.columns, "labels": kmeans.labels_})
    df_centroids = pd.DataFrame(kmeans.cluster_centers_).reset_index()
    if cluster_path_csv != None:
        df_labels.to_csv(cluster_path_csv, index=False)
    else:
        return {"labels": df_labels, "centroids": df_centroids}

def read_cluster_activite(cluster_path_csv):
    """
    Read cluster activite csv

    Parameters
    -------
        cluster_path_csv : String :
            Path to export df_labels DataFrame

    Returns
    -------
        pandas.DataFrame    
    """
    try:
        cluster_activite = pd.read_csv(cluster_path_csv)
        return cluster_activite
    except Exception as e:
        print("Error : can't read cluster activite : ")
        print(e)

def get_cluster_activite(cluster_path_csv, test, train=None):
    """Get cluster activite csv from patch cluster_path_csv.
    Merge cluster with station_id

    Parameters
    ----------
    cluster_path_csv : String :
        Path to export df_labels DataFrame
    test : pandas.DataFrame

    train : pandas.DataFrame

    Returns
    -------

    If train is not None:
        Return 2 pandas.DataFrame train, test
    Else:     
        Return 1 pandas.DataFrame test
    """

    cluster_activite = read_cluster_activite(cluster_path_csv=cluster_path_csv)

    test = test.merge(cluster_activite, left_on='station_id', right_on='id_station',  how='left')
    test.drop('id_station', axis=1, inplace=True)

    if len(train) > 0:
        train = train.merge(cluster_activite, left_on='station_id', right_on='id_station',  how='left')
        train.drop('id_station', axis=1, inplace=True)
        return train, test
    else:
        return test


###################################
###         DATA PROCESS
###################################


def time_resampling(df, freq="10T"):
    """Normalize the timeseries by resampling its timestamps

    Parameters
    ----------
    df : pandas.DataFrame
        Input data, contains columns `ts`, `nb_bikes`, `nb_stands`, `station_id`
    freq : str
        Time resampling frequency

    Returns
    -------
    pandas.DataFrame
        Resampled data
    """

    logger.info("Time resampling for each station by '%s'", freq)
    df = (df.groupby("station_id")
          .resample(freq, on="ts")[["ts", "nb_bikes", "nb_stands", "probability"]]
          .mean()
          .bfill())
    return df.reset_index()

def complete_data(df):
    """Add some temporal columns to the dataset

    - day of the week
    - hour of the day
    - minute

    Parameters
    ----------
    df : pandas.DataFrame
        Input data ; must contain a `ts` column

    Returns
    -------
    pandas.DataFrame
        Data with additional columns `day`, `hour` and `minute`

    """
    logger.info("Complete some data")
    df = df.copy()
    df['day'] = df['ts'].apply(lambda x: x.weekday())
    df['hour'] = df['ts'].apply(lambda x: x.hour)
    df['minute'] = df['ts'].apply(lambda x: x.minute)
    return df

def add_future(df, frequency):
    """Add future bike availability to each observation by shifting input data
    accurate columns with respect to a given `frequency`

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    frequency : DateOffset, timedelta or str
        Indicates the prediction frequency

    Returns
    -------
    pd.DataFrame
        Enriched data, with additional column "future"
    """
    logger.info("Compute the future bike availability (freq='%s')", frequency)
    df = df.set_index(["ts", "station_id"])
    label = df["probability"].copy()
    label.name = "future"
    label = (label.reset_index(level=1)
             .shift(-1, freq=frequency)
             .reset_index()
             .set_index(["ts", "station_id"]))
    logger.info("Merge future data with current observations")
    df = df.merge(label, left_index=True, right_index=True)
    df.reset_index(level=1, inplace=True)
    return df

def prepare_data_for_training(df, date, frequency='1H', start=None, periods=1):
    """Prepare data for training

    Parameters
    ----------
    df : pd.DataFrame
        Input data; must contains a "future" column and a `datetime` index
    date : date.datetime
        Date for the prediction
    frequency : str
        Delay between the training and validation set; corresponds to
    prediction frequency
    start : date.Timestamp
        Start of the history data (for training)
    periods : int
        Number of predictions

    Returns
    -------
    tuple of 4 pandas.DataFrames
        two for training, two for testing (train_X, train_Y, test_X, test_Y)
    """
    logger.info("Split train and test according to a validation date")
    cut = date - pd.Timedelta(frequency.replace('T', 'm'))
    stop = date + periods * pd.Timedelta(frequency.replace('T', 'm'))
    if start is not None:
        df = df[df.index >= start]
    logger.info("Data shape after start cut: %s", df.shape)
    train = df[df.index <= cut].copy()
    logger.info("Data shape after prediction date cut: %s", df.shape)
    train_X = train.drop(["probability", "future"], axis=1)
    train_Y = train['future'].copy()
    # time window
    test = df[df.index >= stop].copy()
    test_X = test.drop(["probability", "future"], axis=1)
    test_Y = test['future'].copy()
    return train_X, train_Y, test_X, test_Y



###################################
###         ALGO
###################################


def fit(train_X, train_Y, test_X, test_Y):
    """Train the xgboost model

    Parameters
    ----------
    train_X : pandas.DataFrame
    test_X : pandas.DataFrame
    train_Y : pandas.DataFrame
    test_Y : pandas.DataFrame

    Returns
    -------
    XGBoost.model
        Booster trained model
    """
    logger.info("Fit training data with the model...")
    # param = {'objective': 'reg:linear'}
    param = {'objective': 'reg:logistic'}
    param['eta'] = 0.2
    param['max_depth'] = 6
    param['silent'] = 1
    param['nthread'] = 4
    param['seed'] = SEED
    training_progress = dict()
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 25
    bst = xgb.train(params=param,
                    dtrain=xg_train,
                    num_boost_round=num_round,
                    evals=watchlist,
                    evals_result=training_progress)
    return bst, training_progress

def train_prediction_model(df, validation_date, frequency):
    """Train a XGBoost model on `df` data with a train/validation split given
    by `predict_date` starting from temporal information (time of the day, day
    of the week) and previous bike availability

    Parameters
    ----------
    df : pandas.DataFrame
        Input data, contains columns `ts`, `nb_bikes`, `nb_stands`,
    `station_id`
    validation_date : datetime.date
        Reference date to split the input data between training and validation
    sets
    frequency : DateOffset, timedelta or str
        Indicates the prediction frequency

    Returns
    -------
    XGBoost.model
        Trained XGBoost model

    """
    df = time_resampling(df)
    df = complete_data(df)
    df = add_future(df, frequency)
    train_test_split = prepare_data_for_training(df,
                                                 validation_date,
                                                 frequency=frequency,
                                                 start=df.index.min(),
                                                 periods=2)
    train_X, train_Y, test_X, test_Y = train_test_split


    # Create cluster activity
    compute_clusters(train_X.reset_index(), cluster_path_csv=CLUSTER_ACT_PATH_CSV)

    # Merge result of cluster activite
    train_X, test_X = get_cluster_activite(CLUSTER_ACT_PATH_CSV, test_X, train_X)



    trained_model = fit(train_X, train_Y, test_X, test_Y)
    return trained_model[0], train_X, train_Y, test_X, test_Y
