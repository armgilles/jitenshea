"""usefull functions
"""

import pandas as pd

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