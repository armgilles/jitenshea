# coding: utf-8

"""Database controller for the Web Flask API
"""


import daiquiri
import logging

from itertools import groupby
from datetime import timedelta
from collections import namedtuple

import pandas as pd

from jitenshea import config
from jitenshea.iodb import db


daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

CITIES = ('bordeaux',
          'lyon')
TimeWindow = namedtuple('TimeWindow', ['start', 'stop', 'order_reference_date'])


def processing_daily_data(rset, window):
    """Re arrange the daily transactions data when it's necessary

    Parameters
    ----------
    rset : sqlalchemy.engine.result.ResultProxy
        Result of a SQL query
    window : integer
        Time windows

    Returns
    -------
    list of dicts
        Rearranged list of transactions

    """
    if not rset:
        return []
    data = [dict(zip(x.keys(), x)) for x in rset]
    if window == 0:
        return data
    # re-arrange the result set to get a list of values for the keys 'date' and 'value'
    values = []
    for k, group in groupby(data, lambda x: x['id']):
        group = list(group)
        values.append({'id': k,
                       "date": [x['date'] for x in group],
                       'value': [x['value'] for x in group],
                       'name': group[0]['name']})
    return {"data": values}

def processing_timeseries(rset):
    """Processing the result of a timeseries SQL query

    Parameters
    ----------
    rset : sqlalchemy.engine.result.ResultProxy
        Result of a SQL query

    Returns
    -------
    a list of dicts
        Bike and stand availability timeseries for given stations

    """
    if not rset:
        return []
    data = [dict(zip(x.keys(), x)) for x in rset]
    values = []
    for k, group in groupby(data, lambda x: x['id']):
        group = list(group)
        values.append({'id': k,
                       'name': group[0]['name'],
                       "ts": [x['ts'] for x in group],
                       'available_bike': [x['available_bike'] for x in group],
                       'available_stand': [x['available_stand'] for x in group]})
    return {"data": values}


def time_window(day, window, backward):
    """Give a start and stop according to the size of `window` and the `backward`
    parameter. The `order_reference_date` is used to fix the date values to sort
    station by values.

    Parameters
    ----------
    day : date
       Start or stop according to the backward parameter
    window : int
       Number of day before (resp. after) the 'day' parameter
    backward : boolean

    Returns
    -------
    TimeWindow
        Ttime between start and stop dates, ordered either by start or stop

    """
    stop = day
    sign = 1 if backward else -1
    start = stop - timedelta(sign * window)
    order_reference_date = stop
    if not backward:
        start, stop = stop, start
        order_reference_date = start
    return TimeWindow(start, stop, order_reference_date)

def station_geojson(stations):
    """Process station data into GeoJSON

    Parameters
    ----------
    stations : list of integer
        Stations IDs

    Returns
    -------
    dict
        Stations into the GeoJSON format
    """
    result = []
    for data in stations:
        result.append(
            {"type": "Feature",
             "geometry": {
                 "type": "Point",
                 "coordinates": [data['x'], data['y']]
             },
             "properties": {
                 "id": data['id'],
                 "name": data['name'],
                 "address": data['address'],
                 "city": data['city'],
                 "nb_bikes": data['nb_bikes']
             }})
    return {"type": "FeatureCollection", "features": result}

def cities():
    """Manually build the list of considered cities

    TODO: recover the number of stations automatically by requesting the database
    # Lyon
    # select count(*) from lyon.pvostationvelov;
    # Bdx
    # select count(*) from bordeaux.vcub_station;

    Returns
    -------
    dict
        List of cities depicted into the API

    """
    return {"data": [{'city': 'lyon',
                      'country': 'france',
                      'stations': 348},
                     {'city': 'bordeaux',
                      'country': 'france',
                      'stations': 174}]}

def stations(city, limit, geojson):
    """Get a set of the `limit` first bicycle stations in `city`; if `geojson` is True, handle GeoJSON format

    Parameters
    ----------
    city : string
    limit : int
    geojson : boolean

    Returns
    -------
    a list of dict
        One dict by bicycle station
    """
    if city == 'bordeaux':
        query = bordeaux_stations(limit)
    elif city == 'lyon':
        query = lyon_stations(limit)
    else:
        raise ValueError("City {} not supported".format(city))
    eng = db()
    rset = eng.execute(query)
    keys = rset.keys()
    result = [dict(zip(keys, row)) for row in rset]
    if geojson:
        return station_geojson(result)
    return {"data": result}


def bordeaux_stations(limit=20):
    """Query for the list of bicycle stations in Bordeaux; give the `limit`
    first ones

    Parameters
    ----------
    limit : int
        Quantity of stations to consider, default 20

    Returns
    -------
    str
        A SQL query to execute
    """
    return """SELECT numstat::int AS id
      ,nom AS name
      ,adresse AS address
      ,lower(commune) AS city
      ,nbsuppor::int AS nb_bikes
      ,st_x(st_transform(geom, 4326)) AS x
      ,st_y(st_transform(geom, 4326)) AS y
    FROM {schema}.vcub_station
    LIMIT {limit}
    """.format(schema=config['bordeaux']['schema'],
               limit=limit)

def lyon_stations(limit=20):
    """Query for the list of bicycle stations in Lyon; give the `limit`
    first ones

    Parameters
    ----------
    limit : int
        Quantity of stations to consider, default 20

    Returns
    -------
    str
        A SQL query to execute
    """
    return """SELECT idstation::int AS id
      ,nom AS name
      ,adresse1 AS address
      ,lower(commune) AS city
      ,nbbornette::int AS nb_bikes
      ,st_x(geom) AS x
      ,st_y(geom) AS y
    FROM {schema}.pvostationvelov
    LIMIT {limit}
    """.format(schema=config['lyon']['schema'],
               limit=limit)

def bordeaux(station_ids):
    """Get some specific bicycle-sharing stations for Bordeaux

    Parameters
    ----------
    station_id : list of int
        Bicycle-sharing station IDs

    Returns
    -------
    list of dict
        Bicycle stations

    """
    query = bordeaux_stations(1).replace("LIMIT 1", 'WHERE numstat IN %(id_list)s')
    eng = db()
    rset = eng.execute(query, id_list=tuple(str(x) for x in station_ids)).fetchall()
    if not rset:
        return []
    return {"data" : [dict(zip(x.keys(), x)) for x in rset]}

def lyon(station_ids):
    """Get some specific bicycle-sharing stations for Lyon

    Parameters
    ----------
    station_id : list of int
        Bicycle-sharing station IDs

    Returns
    -------
    list of dict
        Bicycle stations

    """
    query = lyon_stations(1).replace("LIMIT 1", 'WHERE idstation IN %(id_list)s')
    eng = db()
    rset = eng.execute(query, id_list=tuple(str(x) for x in station_ids)).fetchall()
    if not rset:
        return []
    return {"data" : [dict(zip(x.keys(), x)) for x in rset]}


def station_city_table(city):
    """Name table and ID column name

    Parameters
    ----------
    city : str
        City to consider, either 'bordeaux' or 'lyon'

    Returns
    -------
    tuple of str
        Table and ID column name related to given city

    """
    if city not in ('bordeaux', 'lyon'):
        raise ValueError("City '{}' not supported.".format(city))
    if city == 'bordeaux':
        return 'vcub_station', 'numstat'
    if city == 'lyon':
        return 'pvostationvelov', 'idstation'

def daily_query(city):
    """SQL query to get daily transactions according to `city`

    Parameters
    ----------
    city : str

    Returns
    -------
    str
        SQL request
    """
    if city not in ('bordeaux', 'lyon'):
        raise ValueError("City '{}' not supported.".format(city))
    table, idcol = station_city_table(city)
    return """SELECT id
           ,number AS value
           ,date
           ,Y.nom AS name
        FROM {schema}.daily_transaction AS X
        LEFT JOIN {schema}.{table} AS Y ON X.id=Y.{idcol}::int
        WHERE id IN %(id_list)s AND date >= %(start)s AND date <= %(stop)s
        ORDER BY id,date""".format(schema=config[city]['schema'], table=table, idcol=idcol)

def daily_query_stations(city, limit, order_by='station'):
    """SQL query to get daily transactions for all stations

    Parameters
    ----------
    city : string
        City to consider, either 'bordeaux' or 'lyon'
    limit : integer
        Number of daily transactions to consider
    order_by : str
        Sorting attribute, default is 'station'

    Returns
    -------
    str
        SQL request
    """
    if city not in ('bordeaux', 'lyon'):
        raise ValueError("City '{}' not supported.".format(city))
    if order_by == 'station':
        order_by = 'id'
    if order_by == 'value':
        order_by = 'number DESC'
    table, idcol = station_city_table(city)
    return """WITH station AS (
            SELECT id
              ,row_number() over (partition by null order by {order_by}) AS rank
            FROM {schema}.daily_transaction
            WHERE date = %(order_reference_date)s
            ORDER BY {order_by}
            LIMIT {limit}
            )
        SELECT S.id
          ,D.number AS value
          ,D.date
          ,Y.nom AS name
        FROM station AS S
        LEFT JOIN {schema}.daily_transaction AS D ON (S.id=D.id)
        LEFT JOIN {schema}.{table} AS Y ON S.id=Y.{idcol}::int
        WHERE D.date >= %(start)s AND D.date <= %(stop)s
        ORDER BY S.rank,D.date;""".format(schema=config[city]['schema'],
                                          table=table,
                                          idcol=idcol,
                                          order_by=order_by,
                                          limit=limit)


def daily_transaction(city, station_ids, day, window=0, backward=True):
    """Retrieve the daily transactions for the stations `station_ids` in `city`

    Parameters
    ----------
    stations_ids : list of int
        Station IDs
    day : date
        Date around which transactions must be focused
    window : int
        Number of days to look around (default=0)
    backward : bool
        If True, get data before the date (default=True)

    Returns
    -------
    a list of dicts
        Daily transactions (number of transactions by station and by day) in `city`
    """
    window = time_window(day, window, backward)
    query = daily_query(city)
    eng = db()
    rset = eng.execute(query,
                       id_list=tuple(str(x) for x in station_ids),
                       start=window.start, stop=window.stop).fetchall()
    return processing_daily_data(rset, window)


def daily_transaction_list(city, day, limit, order_by, window=0, backward=True):
    """Retrieve daily transactions for the `limit` first stations in `city`

    Parameters
    ----------
    city : str
        City to consider, either 'bordeaux' or 'lyon'
    day : date
        Date around which transactions must be focused
    limit : int
        Number of transactions to consider
    order_by : str
        Transaction sorting attribute
    window : int
        Number of days to consider around `date` (default=0)
    backward: bool
        If True, get data before the date (default=True)

    Returns
    -------
    a list of dicts
        Daily transactions (number of transactions by station and by day) in `city`

    """
    window = time_window(day, window, backward)
    query = daily_query_stations(city, limit, order_by)
    eng = db()
    rset = eng.execute(query, start=window.start, stop=window.stop,
                       order_reference_date=window.order_reference_date).fetchall()
    return processing_daily_data(rset, window)

def timeseries(city, station_ids, start, stop):
    """Get timeseries data between dates `start` and `stop` for stations
    `station_ids` in `city`

    Parameters
    ----------
    city : str
        City to consider, either 'bordeaux' or 'lyon'
    station_ids : list of integer
        Bike stations to consider
    start : date
        Beginning of the period
    stop : date
        End of the period

    Returns
    -------
    list of dicts
        Bike and stand availability timeseries

    """
    query = """SELECT *
    FROM {schema}.timeserie_norm
    WHERE id IN %(id_list)s AND ts >= %(start)s AND ts < %(stop)s
    """.format(schema=config[city]['schema'])
    eng = db()
    rset = eng.execute(query, id_list=tuple(x for x in station_ids),
                       start=start, stop=stop)
    return processing_timeseries(rset)

def hourly_process(df):
    """Timeseries into a hourly transaction profile

    Parameters
    ----------
    df : pandas.DataFrame
        Timeseries bike data for one specific station

    Returns
    -------
    pandas.DataFrame
        Transactions sum & mean for each hour

    """
    df = df.copy().set_index('ts')
    transaction = (df['available_bike']
                   .diff()
                   .abs()
                   .dropna()
                   .resample('H')
                   .sum()
                   .reset_index())
    transaction['hour'] = transaction['ts'].apply(lambda x: x.hour)
    return transaction.groupby('hour')['available_bike'].agg(['sum', 'mean'])

def hourly_profile(city, station_ids, day, window):
    """Compute the number of transactions per hour

    Note: for `window` parameter, quite annoying to convert np.int64,
    np.float64 from the DataFrame to JSON, even if you convert the DataFrame to
    dict. So, I use the .tolist() np.array method for the index and each
    column.

    Parameters
    ----------
    city : str
        City to consider, either 'bordeaux' or 'lyon'
    station_ids : list of integer
        Shared bike stations IDs
    day : date
        Day around which transactions must be scanned
    window : int
        Number of days to consider around `date`

    Returns
    -------
    a list of dicts
        Per-hour transactions, for given `city` and `station_ids` and dates

    """
    start = day - timedelta(window)
    result = []
    for data in timeseries(city, station_ids, start, day)["data"]:
        df = pd.DataFrame(data)
        profile = hourly_process(df)
        result.append({
            'id': data['id'],
            'name': data['name'],
            'hour': profile.index.values.tolist(),
            'sum': profile['sum'].values.tolist(),
            'mean': profile['mean'].values.tolist()})
    return {"data": result, "date": day, "window": window}


def daily_profile_process(df):
    """DataFrame with dates into a daily transaction profile

    Parameters
    ----------
    df : pandas.DataFrame - bike data timeseries for one specific station

    Returns
    -------
    pandas.DataFrame
        Transactions sum & mean for each day of the week

    """
    df = df.copy()
    df['weekday'] = df['date'].apply(lambda x: x.weekday())
    return df.groupby('weekday')['value'].agg(['sum', 'mean'])

def daily_profile(city, station_ids, day, window):
    """Compute the number of transaction per week day

    Note: for `window` parameter, it is quite annoying to convert np.int64,
    np.float64 from the DataFrame to JSON, even if you convert the DataFrame to
    dict. So, I use the .tolist() np.array method for the index and each
    column.

    Parameters
    ----------
    city : str
    stations_ids : list
    day : date
    window : int
        Number of days

    Returns
    -------
    a list of dicts
        Number of transaction per week day

    """
    result = []
    for data in daily_transaction(city, station_ids, day, window)["data"]:
        df = pd.DataFrame(data)
        profile = daily_profile_process(df)
        result.append({
            'id': data['id'],
            'name': data['name'],
            'day': profile.index.values.tolist(),
            'sum': profile['sum'].values.tolist(),
            'mean': profile['mean'].values.tolist()})
    return {"data": result, "date": day, "window": window}
