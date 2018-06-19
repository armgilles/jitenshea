
import logging
import daiquiri

import numpy as np
import pandas as pd

from jitenshea.armand import (datareader, cleanup, bikes_probability)
from jitenshea.stats import (time_resampling, complete_data, add_future, prepare_data_for_training,
								rmse)

CITY = 'Lyon'
FILE_DATA = 'jitenshea/data/lyon-2018-02-2018-05.csv'

frequency = '3H'

print("In dump_prediction")

import os
os.environ['QT_QPA_PLATFORM']='offscreen' # bug here on dev server

daiquiri.setup(logging.INFO)
logger = daiquiri.getLogger("dump_prediction")


raw = datareader(FILE_DATA)
"""
 number         last_update  bike_stands  available_bike_stands  /
0    1001 2018-03-01 00:01:25           16                    7
1    1001 2018-03-01 00:08:10           16                    6

available_bikes  availabilitycode availability bonus status
0                9                 1         Vert   Non   OPEN
1               10                 1         Vert   Non   OPEN
"""
df_clean = cleanup(raw)
"""
station_id                  ts  nb_stands  nb_bikes status
0        1001 2018-03-01 00:01:25          7         9   OPEN
1        1001 2018-03-01 00:08:10          6        10   OPEN
"""
df = bikes_probability(df_clean)
"""
station_id                  ts  nb_stands  nb_bikes status  probability
0        1001 2018-03-01 00:01:25          7         9   OPEN       0.5625
1        1001 2018-03-01 00:08:10          6        10   OPEN       0.6250

"""

# Calcul validation Date & test date
#Train = 70 % of data set
#Validation = 15 % of data set
#test = 15 % of data set
delta_ts_df = df.ts.max() - df.ts.min()
delta_15_percent = delta_ts_df *0.15
validation_date = df.ts.max() - (delta_15_percent * 2)
test_date = df.ts.max() - (delta_15_percent)

df = time_resampling(df)

df = complete_data(df)

df = add_future(df, frequency)

train_test_split = prepare_data_for_training(df,
                                                 validation_date,
                                                 test_date,
                                                 frequency=frequency,
                                                 start=df.index.min(),
                                                 periods=2)
train_X, train_Y, val_X, val_Y, test_X, test_Y = train_test_split


logger.info("Create last data as target")
train_X = bikes_probability(train_X)
val_X = bikes_probability(val_X)
test_X = bikes_probability(test_X)

rmse_train = rmse(train_Y, train_X['probability'].reset_index(drop=True))
rmse_val = rmse(val_Y, val_X['probability'].reset_index(drop=True))
rmse_test = rmse(test_Y, test_X['probability'].reset_index(drop=True))



print("RMSE train is {}".format(rmse_train))
print("RMSE val is {}".format(rmse_val))
print("RMSE test is {}".format(rmse_test))



