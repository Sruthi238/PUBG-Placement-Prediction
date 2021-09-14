import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import time
start_time = time.time()

train = pd.read_csv('train_V2.csv')
train = train.dropna()

train['total_dist'] = train['swimDistance'] + train['walkDistance'] + train['rideDistance']
train['kills_with_assist'] = train['kills'] + train['assists']
train['headshot_over_kills'] = train['headshotKills'] / train['kills']
train['headshot_over_kills'].fillna(0, inplace=True)

train = train.drop(['Id','groupId','matchId'],axis=1)
matchtype = train.matchType.unique()

match_dict = {}
for i,each in enumerate(matchtype):
	match_dict[each] = i
train.matchType = train.matchType.map(match_dict)

y = train['winPlacePerc']
X = train.drop(['winPlacePerc'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc_X = StandardScaler()
X_trainsc = sc_X.fit_transform(X_train)
X_testsc = sc_X.transform(X_test)

dt = DecisionTreeRegressor()
dt.fit(X_trainsc,y_train)

y_pred_dt = dt.predict(X_testsc)

rmse = sqrt(mean_squared_error(y_test, y_pred_dt))
mse = mean_squared_error(y_test, y_pred_dt)
r2 = r2_score(y_test,y_pred_dt)

print("RMSE = >",rmse)
print("MSE = >",mse)
print("R Squared = >",r2)

res = pd.DataFrame()
res['Actual'] = y_test
res['Predicted'] = y_pred_dt
res['Difference'] = abs(y_test-y_pred_dt)

print(res.head(10))

print("--- %s seconds ---" % (time.time() - start_time))

