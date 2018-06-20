import pytest
import pandas as pd
from datetime import timedelta

from jitenshea import stats
from jitenshea import utils_calendar


# Global variable in pytest
pytest.PATH_TEST_DATA = "tests/data_test/input_test_data.csv"
pytest.BIN_RESAMPLING = "10T"
pytest.FREQ = '3H'


#####################################
#### 			Read data
#####################################
@pytest.fixture
@pytest.mark.pipeline_data
def data():
	"""
	Read test data (original data)
	"""
	data = pd.read_csv(pytest.PATH_TEST_DATA, parse_dates=['ts'])
	return data

@pytest.mark.pipeline_data
def test_read_data_shape(data):
	"""
	Test reading test data shape if no new test set
	"""

	assert data.shape == (758, 6)

#####################################
#### 		data_resampling
#####################################

@pytest.fixture
def data_resampling(data):
	"""
	Get data resampling
	"""
	data_resampling = stats.time_resampling(data)
	return data_resampling

@pytest.mark.pipeline_data
def test_time_resampling(data_resampling):
	"""
	Test time_resampling function in stats.py
	"""

	# is_open features
	assert sorted(data_resampling.is_open.unique()) == [0.0, 1.0]

	# Resampling on '10T'
	list_minute = sorted(data_resampling.ts.dt.minute.unique())
	assert list_minute == [0, 10, 20, 30, 40, 50]

	# DataFrame len (less row with resampling)
	assert len(data_resampling) == 733


#####################################
#### 		data_complete
#####################################

@pytest.fixture
def data_complete(data_resampling):
	"""
	Get data complete
	"""
	data_complete = stats.complete_data(data_resampling)
	return data_complete

@pytest.mark.pipeline_data
def test_complete_data(data_complete):
	"""
	Test complete_data function in stats.py
	"""

	# df shape
	assert data_complete.shape == (733, 9)

	# day feature 
	assert sorted(data_complete.day.unique()) == [2, 3]

	# hour feature
	assert sorted(data_complete.hour.unique()) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

	# minute feature
	assert sorted(data_complete.minute.unique()) == [0, 10, 20, 30, 40, 50]


#####################################
#### 		Advanced features
#####################################

@pytest.mark.pipeline_data
def test_get_public_holiday(data_complete):
	"""
	Test get_public_holiday function in stats.py
	"""

	df = stats.get_public_holiday(data_complete, count_day=5)

	# df shape
	assert df.shape == (733, 11)

	# Bool for public_holiday feature
	assert sorted(df.public_holiday.unique()) == [0, 1]

	# Count day since public_holiday (max 5 by default)
	assert sorted(df.public_holiday_count.unique()) == [0, 1]



@pytest.mark.pipeline_data
def test_get_school_holiday(data_complete):
	"""
	Test get_school_holiday function in stats.py
	"""

	df = stats.get_school_holiday(data_complete)

	# df shape
	assert df.shape == (733, 10)

	# Bool for public_holiday feature
	assert sorted(df.school_holiday.unique()) == [0, 1]

	# create specific DataFrame
	# 2018-01-01 -> 2018-12-30 (364 rows)
	date_df = date_df = pd.DataFrame(pd.date_range('2018-01-01', freq='1D', periods=364), columns=['ts'])

	date_df = stats.get_school_holiday(date_df)

	# Christmas holiday 2017 test
	assert date_df[date_df.ts == "2018-01-02"].school_holiday.values[0] == 1

	# Winter holiday 2018 test
	assert date_df[date_df.ts == "2018-02-12"].school_holiday.values[0] == 1

	# Spring break 2018 test
	assert date_df[date_df.ts == "2018-04-10"].school_holiday.values[0] == 1

	# Summer holiday 2018 test
	assert date_df[date_df.ts == "2018-08-02"].school_holiday.values[0] == 1

	# Random no school holiday
	assert date_df[date_df.ts == "2018-01-28"].school_holiday.values[0] == 0
	assert date_df[date_df.ts == "2018-03-15"].school_holiday.values[0] == 0
	assert date_df[date_df.ts == "2018-06-28"].school_holiday.values[0] == 0
	assert date_df[date_df.ts == "2018-09-28"].school_holiday.values[0] == 0
	assert date_df[date_df.ts == "2018-11-28"].school_holiday.values[0] == 0

@pytest.mark.pipeline_data
def test_create_bool_empty_full_station(data_complete):
	"""
	test create_bool_empty_full_station function in stats.py
	"""


	df = stats.create_bool_empty_full_station(data_complete)

	# df shape
	assert df.shape == (733, 10)

	# Bool for warning_empty_full feature
	assert sorted(df.warning_empty_full.unique()) == [0, 1]

	# Rules
	assert sorted(df[(df.probability >= 0.875) | (df.nb_bikes <= 2)].warning_empty_full.unique()) == [1]


@pytest.mark.pipeline_data
def test_get_station_recently_closed(data_complete):
	"""
	test get_station_recently_closed function in stats.py
	"""

	df = stats.get_station_recently_closed(data_complete)

	# df shape
	assert df.shape == (733, 10)

	# Number of different value for was_recently_open
	assert df.was_recently_open.nunique() == 24

	# Sum of this value
	assert df.was_recently_open.sum() == 17220.0

@pytest.mark.pipeline_data
def test_add_future(data_complete):
	"""
	test add_future function in stats.py
	"""

	df = stats.add_future(data_complete, frequency=pytest.FREQ)

	# df shape (loosing some rows with merge inner)
	assert df.shape == (679, 9)

	df.reset_index(inplace=True)

	if 'H' in pytest.FREQ:
		df['date_futur'] = df.ts + timedelta(hours=int(pytest.FREQ.replace('H', '')))
	elif 'T' in pytest.FREQ:
		df['date_futur'] = df.ts + timedelta(minutes=int(pytest.FREQ.replace('T', '')))
	elif 'D' in pytest.FREQ:
		df['date_futur'] = df.ts + timedelta(days=int(pytest.FREQ.replace('D', '')))

	# Taking the first 400 rows
	for idx, row in df.head(400).iterrows():
		futur_probabilty = row['future']
		station = row['station_id']
		actual_date = row['ts']
		futur_date = row['date_futur']

		# Looking in the DataFrame futur by selection
		assert futur_probabilty ==df[(df.station_id == station) & (df.ts == futur_date)]['probability'].values[0]

@pytest.mark.pipeline_data
def test_create_rolling_mean_features(data_complete):
	"""
	test create_rolling_mean_features function in stats.py
	"""

	df = stats.create_rolling_mean_features(data_complete, 
		                                     features_name='mean_6', 
		                                     feature_to_mean='probability', 
		                                     features_grp='station_id', 
		                                     nb_shift=6)

	# df shape 
	assert df.shape == (733, 10)

	# Test rolling mean value on nb_shift = 6
	assert round(float(df[df.station_id == 5031][190:196]['probability'].mean()), 4) == round(float(df[df.station_id == 5031][195:196]['mean_6']), 4)
	assert round(float(df[df.station_id == 5031][210:216]['probability'].mean()), 4) == round(float(df[df.station_id == 5031][215:216]['mean_6']), 4)

	assert round(float(df[df.station_id == 10012][90:96]['probability'].mean()), 4) == round(float(df[df.station_id == 10012][95:96]['mean_6']), 4)
	assert round(float(df[df.station_id == 10012][110:116]['probability'].mean()), 4) == round(float(df[df.station_id == 10012][115:116]['mean_6']), 4)

	assert round(float(df[df.station_id == 1001][10:16]['probability'].mean()), 4) == round(float(df[df.station_id == 1001][15:16]['mean_6']), 4)
	assert round(float(df[df.station_id == 1001][30:36]['probability'].mean()), 4) == round(float(df[df.station_id == 1001][35:36]['mean_6']), 4)

	# Mean of this value
	assert df.mean_6.mean() == 0.52317199859081198


@pytest.mark.pipeline_data
def test_create_rolling_std_features(data_complete):
	"""
	test create_rolling_std_features function in stats.py
	"""

	df = stats.create_rolling_std_features(data_complete, 
		                                     features_name='std_9', 
		                                     feature_to_std='probability', 
		                                     features_grp='station_id', 
		                                     nb_shift=9)

	# df shape 
	assert df.shape == (733, 10)

	# Test rolling std value on nb_shift = 9
	assert round(float(df[df.station_id == 5031][190:199]['probability'].std()), 4) == round(float(df[df.station_id == 5031][198:199]['std_9']), 4)
	assert round(float(df[df.station_id == 5031][210:219]['probability'].std()), 4) == round(float(df[df.station_id == 5031][218:219]['std_9']), 4)

	assert round(float(df[df.station_id == 10012][90:99]['probability'].std()), 4) == round(float(df[df.station_id == 10012][98:99]['std_9']), 4)
	assert round(float(df[df.station_id == 10012][110:119]['probability'].std()), 4) == round(float(df[df.station_id == 10012][118:119]['std_9']), 4)

	assert round(float(df[df.station_id == 1001][10:19]['probability'].std()), 4) == round(float(df[df.station_id == 1001][18:19]['std_9']), 4)
	assert round(float(df[df.station_id == 1001][30:39]['probability'].std()), 4) == round(float(df[df.station_id == 1001][38:39]['std_9']), 4)

	# Mean of this value
	assert df.std_9.mean() == 0.054640961523453262

