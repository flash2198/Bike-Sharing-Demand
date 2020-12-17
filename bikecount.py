import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xg 
from catboost import CatBoostRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_log_error,mean_absolute_error,mean_squared_error,r2_score,confusion_matrix,classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict


df=pd.read_csv('data/train.csv',parse_dates=['datetime'])
df_t=pd.read_csv('data/test.csv',parse_dates=['datetime'])

df_test=df_t.copy()
df_tmp=df.copy()

df_tmp=df_tmp.sample(frac=1)

# <<------------handling data------------------>>

def preprocess_data(df_func):
	df_func['saleYear']=df_func.datetime.dt.year
	df_func['saleMonth']=df_func.datetime.dt.month
	df_func['saleDay']=df_func.datetime.dt.day
	df_func['saleDayOfWeek']=df_func.datetime.dt.dayofweek
	df_func['saleDayOfYear']=df_func.datetime.dt.dayofyear
	df_func['saleHour']=df_func.datetime.dt.hour


preprocess_data(df_tmp)
preprocess_data(df_test)

df_tmp.drop(['datetime','casual','registered'],axis=1,inplace=True)
df_test.drop('datetime',axis=1,inplace=True)


df_tmp=pd.get_dummies(df_tmp,columns=['season','weather','saleDayOfWeek'])
df_test=pd.get_dummies(df_test,columns=['season','weather','saleDayOfWeek'])


fig,ax=plt.subplots(figsize=(15,10))

ax=sns.heatmap(df_tmp.corr(),annot=True,linewidths=0.5,fmt='.2f',cmap='YlGnBu')

plt.savefig('correlation.png')


for label,content in df_tmp.items():
	if(df_tmp[label].dtype=='float64'):
		df_tmp[label]=df_tmp[label].astype(int)

for label,content in df_test.items():
	if(df_test[label].dtype=='float64'):
		df_test[label]=df_test[label].astype(int)




# <<---------------Splitting data----------------->>


x=df_tmp.drop('count',axis=1)
y=df_tmp['count']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)






# <<----------------Predicting function------------------->>

def predict_me(model):
	print(model)
	y_preds=model.predict(x_test)
	y_preds=np.absolute(y_preds)
	print('model_r^2:',model.score(x_test,y_test))
	print('model_mean_absolute_error:',mean_absolute_error(y_test,y_preds))
	print('model_mean_squared_error:',mean_squared_error(y_test,y_preds))
	print('model_root_mean_squared_error:',np.sqrt(mean_squared_error(y_test,y_preds)))
	print('model_mean_squred_log_error:',mean_squared_log_error(y_test,y_preds))
	print('model_root_mean_squared_log_error:',np.sqrt(mean_squared_log_error(y_test,y_preds)))




# <<---------------------DecisionTreeRegressor---------------------->>

# model=DecisionTreeRegressor()

# ds_grid={"criterion": ["mse", "mae"],
#               "min_samples_split": [10, 20, 40],
#               "max_depth": [2, 6, 8],
#               "min_samples_leaf": [20, 40, 100],
#               "max_leaf_nodes": [5, 20, 100],
#               }

# model.fit(x_train,y_train)

# predict_me(model)

# y_preds=model.predict(df_test)

# y_preds=np.absolute(y_preds)


# df_predict=pd.DataFrame()

# df_predict['datetime']=df_t['datetime']

# df_predict['count']=y_preds

# df_predict.to_csv('predicted.csv',index=False)



 

# <<-----------------------RandomForestRegressor------------------------>>

# model=RandomForestRegressor(n_estimators=150,max_features='auto',max_depth=None)

# model.fit(x_train,y_train)

# predict_me(model)

# y_preds=model.predict(df_test)

# y_preds=np.absolute(y_preds)


# df_predict=pd.DataFrame()

# df_predict['datetime']=df_t['datetime']

# df_predict['count']=y_preds

# df_predict.to_csv('predicted.csv',index=False)

# rf_grid = {"n_estimators": [50,90,100],
#            "max_depth": [None , 10],
#            "min_samples_split":[14,18,6,2],
#            "min_samples_leaf": [1,3,5],
#            "max_features": [0.5, "auto"]}

# rs_model=RandomizedSearchCV(model,param_distributions=rf_grid,n_iter=50,cv=5)

# rs_model.fit(x_train,y_train)

# predict_me(rs_model)



# <<------------------------LinearRegression------------------------------->>

# model=LinearRegression()

# model.fit(x_train,y_train)

# predict_me(model)




# <<---------------------------Lasso---------------------->>

# model=Lasso()

# model.fit(x_train,y_train)

# predict_me(model)




# <<---------------------------Ridge--------------------->>

# model=Ridge()

# model.fit(x_train,y_train)

# predict_me(model)



# <<----------------------------ElasticNet------------------>>

# model=ElasticNet()

# model.fit(x_train,y_train)

# predict_me(model)



# <<-------------GradientBoostingRegressor------------------>>

# model=GradientBoostingRegressor(n_estimators=400,max_depth=5,min_samples_split=2,learning_rate=0.05,loss='ls')

# model.fit(x_train,y_train)

# predict_me(model)



# <<------------------xgboost-------------------->>


# model=xg.XGBRegressor()

# model.fit(x_train,y_train)

# predict_me(model)

# y_preds=model.predict(df_test)

# y_preds=np.absolute(y_preds)


# df_predict=pd.DataFrame()

# df_predict['datetime']=df_t['datetime']

# df_predict['count']=y_preds

# df_predict.to_csv('predicted.csv',index=False)


# <<--------------------SVR--------------------->>

# model=SVR()

# model.fit(x_train,y_train)

# predict_me(model)



# <<--------------------CatBoostRegressor--------------------->>

# model=CatBoostRegressor(learning_rate=0.1)

# model.fit(x_train,y_train)

# predict_me(model)

# y_preds=model.predict(df_test)

# y_preds=np.absolute(y_preds)


# df_predict=pd.DataFrame()

# df_predict['datetime']=df_t['datetime']

# df_predict['count']=y_preds

# df_predict.to_csv('predicted.csv',index=False)

