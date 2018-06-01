
"""Statistical methods used for analyzing the shared bike data
"""

import logging
import daiquiri

import numpy as np
import pandas as pd
from dateutil import parser
from workalendar.europe import France
from sklearn.cluster import KMeans
import xgboost as xgb

from jitenshea import config

daiquiri.setup(logging.INFO)
logger = daiquiri.getLogger("stats")

# French Calendar
cal = France()

SEED = 2018
np.random.seed(SEED)

LYON_STATION_PATH_CSV = 'jitenshea/data/lyon-stations.csv'
CLUSTER_ACT_PATH_CSV ='jitenshea/data/cluster_activite.csv'
CLUSTER_GEO_PATH_CSV ='jitenshea/data/cluster_geo.csv'


###################################
###         CLUSTER ACTIVITE
###################################


### CLUSTER ACTIVITY

def preprocess_data_for_clustering_act(df):
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

def compute_act_clusters(df, cluster_act_path_csv=None):
    """Compute station clusters based on bike availability time series

    Parameters
    ----------
    df : pandas.DataFrame
        Input data, *i.e.* city-related timeseries, supposed to have
    `station_id`, `ts` and `nb_bikes` columns

    cluster_act_path_csv : String :
        Path to export df_labels DataFrame

    Returns
    -------
    If cluster_act_path_csv is not None :
        Return noting (just export DataFrame in cluster_act_path_csv)
    Else :
        dict
            Two pandas.DataFrame, the former for station clusters and the latter
        for cluster centroids


    """
    df_norm = preprocess_data_for_clustering_act(df)
    model = KMeans(n_clusters=4, random_state=SEED)
    kmeans = model.fit(df_norm.T)
    labels = pd.DataFrame({"id_station": df_norm.columns, 
                                "cluster_activty": kmeans.labels_})
    centroids = pd.DataFrame(kmeans.cluster_centers_).reset_index()
    if cluster_act_path_csv != None:
        labels.to_csv(cluster_act_path_csv, index=False)
    else:
        return {"labels": labels, "centroids": centroids}

def read_cluster_activity(cluster_act_path_csv):
    """
    Read cluster activite csv

    Parameters
    -------
        cluster_act_path_csv : String :
            Path to export df_labels DataFrame

    Returns
    -------
        pandas.DataFrame    
    """
    try:
        cluster_activite = pd.read_csv(cluster_act_path_csv)
        return cluster_activite
    except Exception as e:
        print("Error : can't read cluster activite : ")
        print(e)

def get_cluster_activity(cluster_act_path_csv, test, train=None):
    """Get cluster activite csv from patch cluster_path_csv.
    Merge cluster with station_id

    Parameters
    ----------
    cluster_act_path_csv : String :
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

    cluster_activite = read_cluster_activity(cluster_act_path_csv=cluster_act_path_csv)

    test = test.merge(cluster_activite, left_on='station_id', right_on='id_station',  how='left')
    test.drop('id_station', axis=1, inplace=True)

    if len(train) > 0:
        train = train.merge(cluster_activite, left_on='station_id', right_on='id_station',  how='left')
        train.drop('id_station', axis=1, inplace=True)
        return train, test
    else:
        return test


### CLUSTER GEO

def preprocess_data_for_clustering_geo(LYON_STATION_PATH_CSV):
    """Prepare data in order to apply a clustering algorithm

    Parameters
    ----------
    cluster_act_path_csv : String :
        Path to read lyon station csv

    Returns
    -------
    pandas.DataFrame
        Simpified version of `df`, ready to be used for clustering

    """
    
    station = pd.read_csv(LYON_STATION_PATH_CSV)

    return station


def compute_geo_clusters(cluster_geo_path_csv=None):
    """Compute stations clusters based on their geolocalization

    Parameters
    ----------
    cluster_geo_path_csv : String :
        Path to export df_labels DataFrame

    Returns
    ------
    dict
        labels: id station and their cluster id
        centroids: cluster centroids
    """

    station = preprocess_data_for_clustering_geo(LYON_STATION_PATH_CSV)

    X = station[['lat', 'lon']].copy()
    k_means = KMeans(init='k-means++', n_clusters=12).fit(X)
    labels = pd.DataFrame({"id_station": station['idstation'],
                           "cluster_geo": k_means.labels_})
    labels.sort_values(by="id_station", inplace=True)
    centroids = pd.DataFrame(k_means.cluster_centers_, columns=['lat', 'lon'])
    if cluster_geo_path_csv != None:
        labels.to_csv(cluster_geo_path_csv, index=False)
    else:
        return {"labels": labels, "centroids": centroids}


def read_cluster_geo(cluster_geo_path_csv):
    """
    Read cluster activite csv

    Parameters
    -------
        cluster_geo_path_csv : String :
            Path to export labels DataFrame

    Returns
    -------
        pandas.DataFrame    
    """
    try:
        cluster_geo = pd.read_csv(cluster_geo_path_csv)
        return cluster_geo
    except Exception as e:
        print("Error : can't read cluster geo : ")
        print(e)

def get_cluster_geo(cluster_geo_path_csv, test, train=None):
    """Get cluster activite csv from patch cluster_geo_path_csv.
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

    cluster_geo = read_cluster_geo(cluster_geo_path_csv=cluster_geo_path_csv)

    test = test.merge(cluster_geo, left_on='station_id', right_on='id_station',  how='left')
    test.drop('id_station', axis=1, inplace=True)

    if len(train) > 0:
        train = train.merge(cluster_geo, left_on='station_id', right_on='id_station',  how='left')
        train.drop('id_station', axis=1, inplace=True)
        return train, test
    else:
        return test




###################################
###         DATA PROCESS
###################################


def time_resampling(df, freq="10T"):
    """Normalize the timeseries by resampling its timestamps. 
        Transforme "status" into numerical "is_open" Bool

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

    df['is_open'] = 0
    df.loc[df['status'] == "OPEN", 'is_open'] = 1

    df = (df.groupby("station_id")
          .resample(freq, on="ts")[["ts", "nb_bikes", "nb_stands", "is_open", "probability"]]
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

    Set TS on index

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
    logger.info("Data shape after prediction date cut: %s", train.shape)
    train_X = train.drop(["probability", "future"], axis=1)
    train_Y = train['future'].copy()
    # time window
    test = df[df.index >= stop].copy()
    test_X = test.drop(["probability", "future"], axis=1)
    test_Y = test['future'].copy()
    return train_X, train_Y, test_X, test_Y


###################################
###         ADVANCE FEATURES
###################################


def get_summer_holiday(df):
    """
    Create bool for summer holiday

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    df : pandas.DataFrame
    """

    df['date'] = df.ts.dt.date
    df['date'] = df['date'].astype('str')

    # Create DF with unique date (yyyy-mm-dd)
    date_df = pd.DataFrame(df.date.unique(), columns=['date'])
    date_df['date'] = date_df['date'].astype('str')

    date_df['is_holiday'] = date_df['date'].apply(lambda x : parser.parse(x) < parser.parse("2017-09-04"))
    date_df['is_holiday'] = date_df['is_holiday'].astype('int')

    #merging
    df = df.merge(date_df, on='date', how='left')
    df.drop('date', axis=1, inplace=True)
    return df


def get_station_recently_closed(df, nb_hours=4):
    """
    Create a indicator who check the number of periods the station was close during the nb_hours
    - 0 The station was NOT closed during nb_hours
    - > 1 The station was closes X times during nb_hours

    Need to sort the dataframe
    Warning : depend of the périod of resampling

    Parameters
    ----------
    df : pandas.DataFrame
    nb_hours : int - Numbers of hours for the windows analysis


    Returns
    -------
    df : pandas.DataFrame


    """
    # Resorting
    df = df.sort_values(['station_id', 'ts'])

    time_period = nb_hours * 6 # For a 10T resampling, 1 hours -> 6 rows
    df['was_recently_open'] = df['is_open'].rolling(window=time_period, min_periods=1).sum()

    df = df.sort_values(['station_id', 'ts',])

    return df




def create_rolling_mean_features(df, features_name, feature_to_mean, features_grp, nb_shift):
    """
    function to create a rolling mean on "feature_to_mean" called "features_name" 
    groupby "features_grp" on "nb_shift" value
    Have to sort dataframe and re sort at the end
    """
    df.reset_index(inplace=True)
    df = df.sort_values(['station_id', 'ts'])


    # Create rolling features
    df[features_name] = df.groupby(features_grp)[feature_to_mean].apply(lambda x: x.rolling(window=nb_shift, min_periods=1).mean())

    df = df.sort_values(['ts', 'station_id']) 
    df = df.set_index('ts')
    return df


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

    logger.info("Get summer holiday features")
    df = get_summer_holiday(df)

    logger.info("Create recenlty open station indicator")
    df = get_station_recently_closed(df, nb_hours=4)
    
    logger.info("Create Target")
    df = add_future(df, frequency)

    # logger.info("Create mean transformation")
    df = create_rolling_mean_features(df, 
                                         features_name='mean_6', 
                                         feature_to_mean='probability', 
                                         features_grp='station_id', 
                                         nb_shift=6)
    


    logger.info("Split data into train / test dataset")
    train_test_split = prepare_data_for_training(df,
                                                 validation_date,
                                                 frequency=frequency,
                                                 start=df.index.min(),
                                                 periods=2)
    train_X, train_Y, test_X, test_Y = train_test_split

    logger.info("Cluster activity label")
    #Create cluster activity
    compute_act_clusters(train_X.reset_index(), cluster_act_path_csv=CLUSTER_ACT_PATH_CSV)

    # Merge result of cluster activite
    train_X, test_X = get_cluster_activity(CLUSTER_ACT_PATH_CSV, test_X, train_X)

    logger.info("Cluster geo label")
    # Create cluster geo
    compute_geo_clusters(cluster_geo_path_csv=CLUSTER_GEO_PATH_CSV)

    # Merge result of cluster activite
    train_X, test_X = get_cluster_geo(CLUSTER_GEO_PATH_CSV, test_X, train_X)



    trained_model = fit(train_X, train_Y, test_X, test_Y)
    return trained_model[0], train_X, train_Y, test_X, test_Y
