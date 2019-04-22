import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

def Pre(train):
	tt=train.g.values
	train.drop(['g'],axis=1,inplace=True)
	N=[]
	S=[]
	E=[]
	W=[]
	space=[]
	for x in tt:
		sp=x.count(' ')
		e=x.count('\xe4\xb8\x9c')
		s=x.count('\xe5\x8d\x97')
		n=x.count('\xe5\x8c\x97')
		w=x.count('\xe8\xa5\xbf')
		N.append(1.0*n/(n+s+e+w))
		E.append(1.0*e/(n+s+e+w))
		W.append(1.0*w/(n+s+e+w))
		S.append(1.0*s/(n+s+e+w))
		space.append(sp+1)
	train['N']=N
	train['E']=E
	train['S']=S
	train['W']=W
	#train['space']=space
	return train

def Read():
	train=pd.read_csv("/Users/hutianyi/Desktop/train.csv")
	test=pd.read_csv("/Users/hutianyi/Desktop/test.csv")
	train.columns = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s']
	test.columns = ['id','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r']
	id_num= test.id.values
	test.drop(['id'],axis=1,inplace=True)
	train=Pre(train)
	test=Pre(test)
	#test,train=Drop(test,train,'g')
	train=train.fillna(-999)
	test=test.fillna(-999)
	# East=  \xe4\xb8\x9c
	# Sourth=\xe5\x8d\x97
	# North= \xe5\x8c\x97
	#West=   \xe8\xa5\xbf


	x_test=np.array(test).tolist()
	x_train, y_train = featureSet(train)
	return x_test,x_train,y_train,id_num

def featureSet(data):
	yList = data.s.values
	data.drop(['s'],axis=1, inplace=True)
	xList= np.array(data).tolist()
	return xList, yList

def Cross_validation(xgtrain):
	xgb_params={
		'silent': 1,
		'max_depth': 5
		
	}
	#cvresult = xgb.cv(xgb_params, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=10,metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
	print 'Training Start'
	cvresult = xgb.cv(xgb_params, xgtrain,nfold=10,metrics='rmse',
                   show_stdv=False)
	print cvresult
	return xgb_params

def Train(x_train,y_train):
	print 'Train'
	model = xgb.XGBRegressor(max_depth=15, min_child_weight = 3)
	model.fit(x_train,y_train)
	return model

def Predict(model,x_test,id_num):
	ans = model.predict(x_test)
	with open('/Users/hutianyi/Desktop/res.csv','w') as f:
		f.write('id,price\n')
		x=len(ans)
		for i in range(0,x):
			f.write(str(id_num[i])+','+str(ans[i])+'\n')

if __name__ == '__main__':
	x_test,x_train,y_train,id_num=Read()
	xgtrain = xgb.DMatrix(x_train, y_train)

	xgb_params=Cross_validation(xgtrain)
	#model=Train(x_train,y_train)
	#Predict(model,x_test,id_num)