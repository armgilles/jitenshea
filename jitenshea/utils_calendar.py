
"""Utils for calendar creation (school holiday)
"""

import logging
import daiquiri

import pandas as pd
from datetime import date, timedelta

import os
os.environ['QT_QPA_PLATFORM']='offscreen' # bug here on dev server


daiquiri.setup(logging.INFO)
logger = daiquiri.getLogger("utils_calendar")


URL = "https://www.schoolholidayseurope.eu/france.html"

mapping_city_zone = {'Bordeaux' : 'Zone A', 
					 'Lyon' : 'Zone A',
					 'Nantes' : 'Zone B',
					 'Marseille' : 'Zone B',
					 'Paris' : 'Zone C', 
					 'Toulouse' : 'Zone C', 
					 }

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')


def get_zone_by_city(city_name='Lyon'):
	"""

	"""

	logger.info("Get holiday zone for '%s'", city_name)

	try:
		zone = mapping_city_zone[city_name]
	except:
		try:
			zone = mapping_city_zone[city_name.title()]
		except:
			logger.warning("Can't find zone for '%s', check on mapping_city_zone", city_name)
			zone = 'error'

	return zone

def get_holiday(zone):
	"""

	"""
	try:
		#df_holiday = pd.read_html(URL, match="Region", parse_dates=['Start date', 'End date'])[0]
		df_holiday = pd.read_html(URL, match="Region")[0]
		#df_holiday = pd.read_html(URL, match="Region", parse_dates=['Start date', 'End date'], attrs = {'class':"zebra"})[0]

		logger.info("Load holiday's data")
		export_holiday(df_holiday)
	except ValueError:
		# Error sometime happens (slow website ?)
		logger.warning("Can't load holiday's data. Going to load csv file backup")
		df_holiday = read_holiday()

	df_holiday['Start date'] = df_holiday['Start date'].apply(lambda x: pd.datetime.strptime(x, '%d-%m-%Y'))
	df_holiday['End date'] = df_holiday['End date'].apply(lambda x: pd.datetime.strptime(x, '%d-%m-%Y'))

	zone_letter = zone[-1]
	zone_holiday = df_holiday[df_holiday.Region.str.contains(zone_letter)]

	return zone_holiday

def export_holiday(df_holiday):
	"""

	"""
	try:
		df_holiday.to_csv('jitenshea/data/holiday.csv', index=False)
	except:
		logger.warning("Find no backup holiday's data")

def read_holiday():
	"""

	"""
	df_holiday = pd.read_csv('jitenshea/data/holiday.csv')
	return df_holiday


def create_extra_holiday(zone_holiday):
	"""

	"""
	list_extra_holiday = []
	for idx, rows in zone_holiday.iterrows():
		begin = rows['Start date'].date()
		end = rows['End date'].date()
		delta = end - begin

		for i in range(delta.days + 1):
			list_extra_holiday.append(begin + timedelta(i))


	return list_extra_holiday


# zone = get_zone_by_city('Lyon')
# zone_holiday = get_holiday(zone)

# list_extra_holiday = create_extra_holiday(zone_holiday)








