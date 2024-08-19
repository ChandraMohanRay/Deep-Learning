#Video 10. Practical No:5(a) 
#Aim: Evaluating feed forward deep network for regression using KFold cross validation. 

import pandas as pd 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.wrappers.scikit_learn import KerasRegressor 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline 

dataframe=pd.read_csv("housing.csv",delim_whitespace=True,header=None) 
dataset=dataframe.values 
X=dataset[:,0:] 
Y=dataset[:,:] 
def wider_model(): 
  model=Sequential() 
  model.add(Dense(15,input_dim=13,kernel_initializer='normal',activation='relu')) 
  model.add(Dense(13,kernel_initializer='normal',activation='relu')) 
  model.add(Dense(1,kernel_initializer='normal')) 
  model.compile(loss='mean_squared_error',optimizer='adam') 
  return model 
estimators=[] 
estimators.append(('standardize',StandardScaler())) 
estimators.append(('mlp',KerasRegressor(build_fn=wider_model,epochs=100,batch_size=5))) 
pipeline=Pipeline(estimators) 
kfold=KFold(n_splits=10) 
results=cross_val_score(pipeline,X,Y,cv=kfold) 
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))