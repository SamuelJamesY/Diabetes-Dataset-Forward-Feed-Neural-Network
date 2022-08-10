'''
Forward Feed Neural Network Sklearn for Pima Diabetes dataset
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
import random 


def load_data():
	'''
	Load the data, scale the data, give helpful column names
	'''
	df = pd.read_csv('pima.csv',header=None)
	df.columns = ['preg','glucouse','bloodpressure','skinthickness','insulin','bmi','dbf','age','diabetes']
	scaler = MinMaxScaler()
	df.iloc[:,0:8] = scaler.fit_transform(df.iloc[:,0:8])
	return df

def train_test_data_split(data_frame,expnum):
	'''
	Split the data into train and test set (60/40) and have random_state based on experiment number
	'''
	X = data_frame.iloc[:,0:8].to_numpy()
	Y = data_frame.iloc[:,8].to_numpy()
	xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.4,random_state=expnum)
	return xtrain,xtest,ytrain,ytest

def neural_net(xtrain,xtest,ytrain,ytest,hidden,lr,modeltype,expnum):
	'''
	Build two neural network models, one using an adam optimiser, one using an sgd optimiser
	'''
	if modeltype == 0:
		nn = MLPClassifier(hidden_layer_sizes =(hidden,hidden),solver='adam',max_iter=500,learning_rate_init=lr,random_state=expnum)
	else: 
		nn = MLPClassifier(hidden_layer_sizes =(hidden,hidden),solver='sgd',max_iter=500,learning_rate_init=lr,random_state=expnum)
	nn.fit(xtrain,ytrain)
	ypred = nn.predict(xtest) # predict the y value
	acc_score = accuracy_score(ypred,ytest) # calculate the accuracy score
	return acc_score

def main():
	lr = 0.01
	modeltype = 0
	df = load_data()
	# this takes the mean of 10 experiments for accuracy score for 10 differnt hidden node cases
	acc_score_mean_list=np.empty(10)
	acc_score_std_list=np.empty(10)
	hidden_lst = list(range(6,26,2))
	for hid in hidden_lst:
		acc_score_list = np.empty(10)
		for exp in range(10):
			xtrain,xtest,ytrain,ytest = train_test_data_split(df,exp)
			acc_score_list[exp] = neural_net(xtrain,xtest,ytrain,ytest,hid,lr,modeltype,exp)
		acc_score_mean_list[hidden_lst.index(hid)] = acc_score_list.mean()
		acc_score_std_list[hidden_lst.index(hid)] = acc_score_list.std()
	print('Hidden nodes tested',hidden_lst)
	print('Mean for different hidden nodes',acc_score_mean_list)
	print('Std for different hidden nodes',acc_score_std_list)

if __name__ == '__main__':
	main()