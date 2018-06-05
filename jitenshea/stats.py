
"""Statistical methods used for analyzing the shared bike data
"""

import logging
import daiquiri

import numpy as np
import pandas as pd
from dateutil import parser
from datetime import timedelta
from workalendar.europe import France
from sklearn.cluster import KMeans
import xgboost as xgb

from jitenshea import config

daiquiri.setup(logging.INFO)
logger = daiquiri.getLogger("stats")

# French Calendar
cal = France()

# Reproductivity
SEED = 2018
np.random.seed(SEED)

LYON_STATION_PATH_CSV = 'jitenshea/data/lyon-stations.csv'
CLUSTER_ACT_PATH_CSV ='jitenshea/data/cluster_activite.csv'
CLUSTER_GEO_PATH_CSV ='jitenshea/data/cluster_geo.csv'


# Xgboost parameter
XGB_PARAM ={"objective": "reg:logistic",
              "booster" : "gbtree",
              "eta": 0.2,
              "max_depth": 8,
              "subsample":0.9,
              "silent": 1,
              "seed": SEED}
NUM_ROUND = 40


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
    Read cluster activity csv

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

def get_cluster_activity(cluster_act_path_csv, val, test, train=None):
    """Get cluster activite csv from patch cluster_path_csv.
    Merge cluster with station_id

    Parameters
    ----------
    cluster_act_path_csv : String :
        Path to export df_labels DataFrame
    val : pandas.DataFrame
    test : pandas.DataFrame
    train : pandas.DataFrame

    Returns
    -------

    If train is not None:
        Return 3 pandas.DataFrame train, val, test
    Else:     
        Return 2 pandas.DataFrame val, test
    """

    cluster_activite = read_cluster_activity(cluster_act_path_csv=cluster_act_path_csv)

    val = val.merge(cluster_activite, left_on='station_id', right_on='id_station',  how='left')
    val.drop('id_station', axis=1, inplace=True)

    test = test.merge(cluster_activite, left_on='station_id', right_on='id_station',  how='left')
    test.drop('id_station', axis=1, inplace=True)

    if len(train) > 0:
        train = train.merge(cluster_activite, left_on='station_id', right_on='id_station',  how='left')
        train.drop('id_station', axis=1, inplace=True)
        return train, val, test
    else:
        return val, test


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

def get_cluster_geo(cluster_geo_path_csv, val, test, train=None):
    """Get cluster activite csv from patch cluster_geo_path_csv.
    Merge cluster with station_id

    Parameters
    ----------
    cluster_path_csv : String :
        Path to export df_labels DataFrame
    val : pandas.DataFrame
    test : pandas.DataFrame
    train : pandas.DataFrame

    Returns
    -------

    If train is not None:
        Return 3 pandas.DataFrame train, val, test
    Else:     
        Return 2 pandas.DataFrame val, test
    """

    cluster_geo = read_cluster_geo(cluster_geo_path_csv=cluster_geo_path_csv)

    val = val.merge(cluster_geo, left_on='station_id', right_on='id_station',  how='left')
    val.drop('id_station', axis=1, inplace=True)

    test = test.merge(cluster_geo, left_on='station_id', right_on='id_station',  how='left')
    test.drop('id_station', axis=1, inplace=True)

    if len(train) > 0:
        train = train.merge(cluster_geo, left_on='station_id', right_on='id_station',  how='left')
        train.drop('id_station', axis=1, inplace=True)
        return train, val, test
    else:
        return val, test




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

def prepare_data_for_training(df, validation_date, test_date, frequency='1H', start=None, periods=1):
    """Prepare data for training

    Parameters
    ----------
    df : pd.DataFrame
        Input data; must contains a "future" column and a `datetime` index
    validation_date : date.datetime
        Date for cut the train set into validation set
    test_date : date.datetime
        Date for cut the train set into test set 
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
        two for training, two for testing (train_X, train_Y, val_X, val_Y, test_X, test_Y)
    """
    logger.info("Split train and test according to a validation date")
    cut = validation_date - pd.Timedelta(frequency.replace('T', 'm'))
    stop = validation_date + periods * pd.Timedelta(frequency.replace('T', 'm'))
    if start is not None:
        df = df[df.index >= start]
    logger.info("Data shape after start cut: %s", df.shape)
    train = df[df.index <= validation_date].copy()
    logger.info("Data shape after prediction date cut: %s", train.shape)
    train_X = train.drop(["probability", "future"], axis=1)
    train_Y = train['future'].copy()
    

    # Splitting dateset into validation and test set on time windows
    val = df[(df.index > validation_date) & (df.index <= test_date)].copy()
    val_X = val.drop(["probability", "future"], axis=1)
    val_Y = val['future'].copy()

    test = df[df.index > test_date].copy()
    test_X = test.drop(["probability", "future"], axis=1)
    test_Y = test['future'].copy()

    logger.info("Train min date : %s / max date : %s - %s in total for %s rows", train_X.index.min(), train_X.index.max(), train_X.index.max() - train_X.index.min(), len(train_X))
    logger.info("Valdation min date : %s / max date : %s - %s in total for %s rows", val_X.index.min(), val_X.index.max(), val_X.index.max() - val_X.index.min(), len(val_X))
    logger.info("Test min date : %s / max date : %s - %s in total for %s rows", test_X.index.min(), test_X.index.max(), test_X.index.max() - test_X.index.min(), len(test_X))
    return train_X, train_Y, val_X, val_Y, test_X, test_Y


###################################
###         ADVANCE FEATURES
###################################


def get_public_holiday(df, count_day=None):
    """
    Calcul delta with the closest holiday (count_day before and after) on absolute

    Parameters
    ----------
    df : pandas.DataFrame
    count_day : int : number of day we look for before and after a holiday

    Returns
    -------
    df : pandas.DataFrame

    """
    df['date'] = df.ts.dt.date
    df['date'] = df['date'].astype('str')

    # Create DF with unique date (yyyy-mm-dd)
    date_df = pd.DataFrame(df.date.unique(), columns=['date'])
    date_df['date'] = date_df['date'].astype('str')
    # Create bool
    date_df['public_holiday'] = date_df.date.apply(lambda x: cal.is_holiday(parser.parse(x)))
    date_df['public_holiday'] = date_df['public_holiday'].astype(int)
    
    # Calcul the delta between the last public_holiday == 1 (max count_day)
    if count_day is not None:
        logger.info("compute delta with  public holiday on '%s' days", count_day)
        dt_list = []
        for holyday_day in date_df[date_df.public_holiday == 1].date.unique():
            for i in range(-count_day, count_day+1, 1):
                new_date = parser.parse(holyday_day) + timedelta(days=i)
                new_date_str = new_date.strftime("%Y-%m-%d")
                dt_list.append({'date' : new_date_str,
                                'public_holiday_count' : np.abs(i)})
        # DataFrame
        df_date_count = pd.DataFrame(dt_list)
        # Merging
        date_df = date_df.merge(df_date_count, on='date', how='left')
        # Filling missing value
        date_df['public_holiday_count'] = date_df['public_holiday_count'].fillna(0)

    #merging
    df = df.merge(date_df, on='date', how='left')
    df.drop('date', axis=1, inplace=True)
    return df

def create_bool_empty_full_station(df):
    """
    Create a bool features "warning_empty_full"
    If nb_bike <= 2 --> 1
    If Proba >= 0.875 --> 1
    else --> 0

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    df : pandas.DataFrame
    """
    
    df['warning_empty_full'] = 0
    df.loc[df['nb_bikes'] <= 2, 'warning_empty_full'] = 1
    df.loc[df['probability'] >= 0.875, 'warning_empty_full'] = 1
    
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




def create_rolling_mean_features(df, features_name, feature_to_mean, features_grp, nb_shift, reset_index=True):
    """
    function to create a rolling mean on "feature_to_mean" called "features_name" 
    groupby "features_grp" on "nb_shift" value
    Have to sort dataframe and re sort at the end

    Parameters
    ----------

    df : pandas.DataFrame
    features_name : string - Name of the new feature
    feature_to_median : string - Name of the feature to aggregate
    features_grp : string - Name of the feature to groupby
    nb_shift : int - Number of point to use to rolling aggregate
    reset_index : Bool (opt) - To reset_index in the beginning of the function and at the end

    Returns
    -------

    df : pandas.DataFrame
    """

    if reset_index==True:
        df.reset_index(inplace=True)
    df = df.sort_values(['station_id', 'ts'])


    # Create rolling features
    df[features_name] = df.groupby(features_grp)[feature_to_mean].apply(lambda x: x.rolling(window=nb_shift, min_periods=1).mean())

    df = df.sort_values(['ts', 'station_id'])
    if reset_index==True:
        df = df.set_index('ts')
    return df

def create_rolling_std_features(df, features_name, feature_to_std, features_grp, nb_shift, reset_index=True):
    """
    function to create a rolling std on "feature_to_std" called "features_name" 
    groupby "features_grp" on "nb_shift" value
    Have to sort dataframe and re sort at the end

    Parameters
    ----------

    df : pandas.DataFrame
    features_name : string - Name of the new feature
    feature_to_median : string - Name of the feature to aggregate
    features_grp : string - Name of the feature to groupby
    nb_shift : int - Number of point to use to rolling aggregate
    reset_index : Bool (opt) - To reset_index in the beginning of the function and at the end

    Returns
    -------

    df : pandas.DataFrame
    """
    if reset_index==True:
        df.reset_index(inplace=True)
    df = df.sort_values(['station_id', 'ts'])


    # Create rolling features
    df[features_name] = df.groupby(features_grp)[feature_to_std].apply(lambda x: x.rolling(window=nb_shift, min_periods=1).std())

    df = df.sort_values(['ts', 'station_id']) 
    if reset_index==True:
        df = df.set_index('ts')
    return df

def create_rolling_median_features(df, features_name, feature_to_median, features_grp, nb_shift, reset_index=True):
    """
    function to create a rolling median on "feature_to_median" called "features_name" 
    groupby "features_grp" on "nb_shift" value
    Have to sort dataframe and re sort at the end

    Parameters
    ----------

    df : pandas.DataFrame
    features_name : string - Name of the new feature
    feature_to_median : string - Name of the feature to aggregate
    features_grp : string - Name of the feature to groupby
    nb_shift : int - Number of point to use to rolling aggregate
    reset_index : Bool (opt) - To reset_index in the beginning of the function and at the end

    Returns
    -------

    df : pandas.DataFrame
    """
    if reset_index==True:
        df['ts'] = df.index
    df = df.sort_values(['station_id', 'ts'])


    # Create rolling features
    df[features_name] = df.groupby(features_grp)[feature_to_median].apply(lambda x: x.rolling(window=nb_shift, min_periods=1).median())

    df = df.sort_values(['ts', 'station_id'])
    if reset_index==True:
        df = df.set_index('ts')
    return df


def create_ratio_filling_bike_on_bike(df, cluster_name):
    """
    Get filling bike station on station cluster_name (geo or activity)
    Calcul number of total stand on cluster_name / time
    Calcul number of bike on cluster_name / time
    Create ratio on total nb_stand and nb_bike on station on cluster_name
    Merge the result with the DataFrame

    Parameters
    ----------

    df : pandas.DataFrame
    cluster_name : string - Name of the cluster to groupby

    Returns
    -------

    df : pandas.DataFrame
    """

    # Total stand for station
    df['total_stand'] = df['nb_bikes'] + df['nb_stands']

    # Total stand by time and geo cluster
    total_stand_by_geo_cluster = df.groupby(['ts', cluster_name], as_index=False)['total_stand'].sum()
    total_stand_by_geo_cluster.rename(columns={'total_stand':'total_stand_'+cluster_name}, inplace=True)

    # Total bike by time and geo cluster on nb_bike
    features_by_geo_cluster = df.groupby(['ts', cluster_name], as_index=False)["nb_bikes"].sum()
    features_by_geo_cluster.rename(columns={"nb_bikes":"nb_bikes_"+ cluster_name}, inplace=True)

    # Merging this 2 DataFrame
    grp_features_geo_cluster = total_stand_by_geo_cluster.merge(features_by_geo_cluster, 
                                                                on=['ts', cluster_name], 
                                                                how='inner')

    # Create Ratio
    grp_features_geo_cluster['ratio_nb_bikes_'+cluster_name] = grp_features_geo_cluster['nb_bikes_'+cluster_name] / grp_features_geo_cluster['total_stand_' + cluster_name]
    grp_features_geo_cluster = grp_features_geo_cluster[['ts', cluster_name, 'ratio_nb_bikes_'+cluster_name]]
    # Merge with df
    df = df.merge(grp_features_geo_cluster, on=['ts', cluster_name], how='left')
    #df = df.drop('total_stand', axis=1)
    return df


def interaction_features(a, b, df):
    """
    Create interaction between 2 features (a and b)
    Return :
     - Minus (a-b)
     - multiply (a*b)
     - ratio (a/b)    
    """
    
    ## Minus
    minus_label = a+'_minus_'+b
    df[minus_label] = df[a] - df[b]
    
    ## Multiply
    milty_label = a+'_multi_'+b
    df[milty_label] = df[a] * df[b]
    
    ## Ratio 
    ratio_label = a+'_ratio_'+b
    df[ratio_label] = df[a] / df[b]

    # For inf value
    df[ratio_label] = df[ratio_label].replace(np.inf, 0)
    
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
    # param = {'objective': 'reg:logistic'}
    # param['eta'] = 0.2
    # param['max_depth'] = 6
    # param['silent'] = 1
    # param['nthread'] = 4
    # param['seed'] = SEED
    training_progress = dict()
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    bst = xgb.train(params=XGB_PARAM,
                    dtrain=xg_train,
                    num_boost_round=NUM_ROUND,
                    evals=watchlist,
                    evals_result=training_progress)
    return bst, training_progress


def rmse(y_true, y_pred):
        """Compute RMSE score
    Parameters
    ----------
    y_true : pandas.Serie
    y_pred : pandas.Serie

    Returns
    -------
    rmse : Float
    """
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        return rmse

def score_model(model, test_X, test_Y):
    """Predict on test_X with model 
            Score the prediction with reality (test_Y) RMSE
    Parameters
    ----------
    model : Xgboost model
    test_X : pandas.DataFrame
        test set with features
    test_Y : pandas.Series

    Returns
    -------
    None
    """
    y_pred = model.predict(xgb.DMatrix(test_X))
    rmse_score = rmse(test_Y, y_pred)
    
    print("RMSE score model on Test set is {}".format(rmse_score))


def train_prediction_model(df, validation_date, test_date, frequency):
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
    test_date : datetime.date
        Reference date to split the input data between training and test sets
    frequency : DateOffset, timedelta or str
        Indicates the prediction frequency

    Returns
    -------
    XGBoost.model
        Trained XGBoost model

    """
    df = time_resampling(df)
    df = complete_data(df)

    # logger.info("Get summer holiday features")
    # df = get_summer_holiday(df)

    logger.info("Get public holiday features")
    df = get_public_holiday(df, count_day=5)

    logger.info("create bool empty full station")
    df = create_bool_empty_full_station(df)

    logger.info("Create recenlty open station indicator")
    df = get_station_recently_closed(df, nb_hours=4)
    
    logger.info("Create Target")
    df = add_future(df, frequency)

    logger.info("Create mean transformation")
    df = create_rolling_mean_features(df, 
                                     features_name='mean_6', 
                                     feature_to_mean='probability', 
                                     features_grp='station_id', 
                                     nb_shift=6)

    logger.info("Create std transformation")
    df = create_rolling_std_features(df, 
                                     features_name='std_9', 
                                     feature_to_std='probability', 
                                     features_grp='station_id', 
                                     nb_shift=9)

    # logger.info("Create median transformation")
    # df = create_rolling_median_features(df, 
    #                                      features_name='median_6', 
    #                                      feature_to_median='probability', 
    #                                      features_grp='station_id', 
    #                                      nb_shift=6)


    # logger.info("Create interaction features with 'mean_6' and 'median_6' ")
    # df = interaction_features('mean_6', 'median_6', df)

    logger.info("Split data into train / test dataset")
    train_test_split = prepare_data_for_training(df,
                                                 validation_date,
                                                 test_date,
                                                 frequency=frequency,
                                                 start=df.index.min(),
                                                 periods=2)
    train_X, train_Y, val_X, val_Y, test_X, test_Y = train_test_split

#     train_X.tail()
#                         station_id  nb_bikes  nb_stands  is_open  day  hour  \
# ts
# 2018-05-11 23:30:00       11001       1.0       17.0      1.0    4    23
# 2018-05-11 23:30:00       11002       5.0       15.0      1.0    4    23

    # Keep TS's index in memory
    train_index = train_X.index
    val_index = val_X.index
    test_index = test_X.index

    logger.info("Cluster activity label")
    #Create cluster activity
    compute_act_clusters(train_X.reset_index(), cluster_act_path_csv=CLUSTER_ACT_PATH_CSV)
    # Merge result of cluster activite
    train_X, val_X, test_X = get_cluster_activity(CLUSTER_ACT_PATH_CSV, val_X, test_X, train_X)

    logger.info("Cluster geo label")
    # Create cluster geo
    compute_geo_clusters(cluster_geo_path_csv=CLUSTER_GEO_PATH_CSV)
    # Merge result of cluster activite
    train_X, val_X, test_X = get_cluster_geo(CLUSTER_GEO_PATH_CSV, val_X, test_X, train_X)

    # Give back TS index in train_X, val_X & test_X
    train_X.set_index(train_index, inplace=True)
    val_X.set_index(val_index, inplace=True)
    test_X.set_index(test_index, inplace=True)


    # Have to concat X_train & X_test to calculate rolling windows ratio
    n_train = len(train_X)
    n_val = len(val_X)
    n_test = len(test_X)
    df_all = pd.concat([train_X, val_X, test_X])

    # df_all.tail()
    #                      station_id  nb_bikes  nb_stands  is_open  day  hour  \
    # ts
    # 2018-05-29 23:20:00       10120       2.0       12.0      1.0    1    23
    # 2018-05-29 23:20:00       10121      11.0        5.0      1.0    1    23
    #              minute  is_holiday  was_recently_open    mean_6  \
    # ts
    # 2018-05-29 23:20:00      20           0               24.0  0.130952
    # 2018-05-29 23:20:00      20           0               24.0  0.677083
    #            std_9  cluster_activty  cluster_geo
    # ts
    # 2018-05-29 23:20:00  0.029161                2            4
    # 2018-05-29 23:20:00  0.032940                2            4


    # Ts index have to be use for create ratio to cluster
    df_all = df_all.reset_index()

    logger.info("Create Ratio of bike dispo on cluster geo")
    df_all = create_ratio_filling_bike_on_bike(df_all, 'cluster_geo')

    logger.info("Create Ratio of bike dispo on cluster activity")
    df_all = create_ratio_filling_bike_on_bike(df_all, 'cluster_activty')

    logger.info("Create mean transformation on ratio of bike on cluster geo")
    df_all = create_rolling_mean_features(df_all, 
                                     features_name='ratio_nb_bikes_cluster_geo_mean_6', 
                                     feature_to_mean='ratio_nb_bikes_cluster_geo', 
                                     features_grp='station_id', 
                                     nb_shift=6,
                                     reset_index=False)

    logger.info("Create std transformation on ratio of bike on cluster activity")
    df_all = create_rolling_std_features(df_all, 
                                     features_name='ratio_nb_bikes_cluster_activty_std_6', 
                                     feature_to_std='ratio_nb_bikes_cluster_activty', 
                                     features_grp='station_id', 
                                     nb_shift=6,
                                     reset_index=False)

    # Cut df_all to give back train_X & test_X
    train_X = df_all[: n_train].copy()
    val_X = df_all[n_train: n_train + n_val].copy()
    test_X = df_all.tail(n_test).copy()


    train_X.drop('ts', axis=1, inplace=True)
    val_X.drop('ts', axis=1, inplace=True)
    test_X.drop('ts', axis=1, inplace=True)


    trained_model = fit(train_X, train_Y, val_X, val_Y)
    model = trained_model[0]

    score_model(model, test_X, test_Y)

    return trained_model[0], train_X, train_Y, val_X, val_Y, test_X, test_Y
