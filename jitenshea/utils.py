"""usefull functions
"""

import pandas as pd
from datetime import datetime

import xgboost as xgb

def get_features_importance(clf, top=24):
	"""
	To plot features importance fscore

	Parameters
    ----------
    clf : Xgboost model
	top : int (opt) - numbers of features ot display

	Returns
    -------
    pandas.DataFrame
	"""

	list_importance = []

	for var in clf.get_fscore().items():
		list_importance.append({'col_name' : var[0],
								'importance' : var[1]})

	df_importance = pd.DataFrame(list_importance)

	df_importance = df_importance.sort_values('importance', ascending=0).head(top)
	return df_importance

def export_data_set(train_X, train_Y, val_X, val_Y, test_X, test_Y, train_index, val_index, test_index, model, export_name=None):
	train = train_X.reset_index(drop=True).copy()
	train['y_pred'] = model.predict(train_X)
	train['y'] = train_Y
	train.set_index(train_index, inplace=True)
	train['type'] = 'train'

	val = val_X.reset_index(drop=True).copy()
	val['y_pred'] = model.predict(val_X)
	val['y'] = val_Y
	val.set_index(val_index, inplace=True)
	val['type'] = 'val'

	test = test_X.reset_index(drop=True).copy()
	test['y_pred'] = model.predict(test_X)
	test['y'] = test_Y
	test.set_index(test_index, inplace=True)
	test['type'] = 'test'

	if export_name == None:
		export_name =  datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

	data = pd.concat([train, val, test])
	path_to_data = 'jitenshea/data/analyse_data_'
	data.to_csv(path_to_data+export_name+'.csv')


