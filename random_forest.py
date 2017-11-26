import numpy as np
import os
import sys
import csv
import math
import types
import sklearn
from sklearn.preprocessing import scale

#from sklearn import cross_validation
from sklearn.model_selection import ShuffleSplit
import random


#sys.setrecursionlimit(2000)
def loadData(filename):
	X = []
	Y = []
	with open(filename) as f:
		content = f.readlines()
	for item in content:
		X.append(item.split()[1:])
		Y.append(item.split()[0])
	return X, Y

def split_in_folds(X,Y,folds):#for testing 
	#print len(X)
	fold_size=len(X)/folds
	#print fold_size
	fold_ratio=fold_size/float(len(X))
	test_size=fold_ratio
	#print "test_size",test_size
	rs = ShuffleSplit(n_splits=folds, test_size=fold_ratio, random_state=0)
	lis = []
	for train_index, test_index in rs.split(X):
		lis.append([train_index, test_index])
	#print lis

	return lis #has indexes

def tree_split_right_left(index,value_of_feature,X,Y):
	left = list()
	right = list()
	count=0
	for row in X:
		if row[index]<0:
			#print row[index],value_of_feature
			#print Y[count]
			left.append([row,Y[count]])
			#print left
		else:
			right.append([row,Y[count]])
		count=count+1
	#print len(left)
	return left,right

def gini_index(groups, label_set):
	gini=0
	for label in label_set:
		label_count=0
		for group in groups:#left and right
			size_lr=len(group)
			if size_lr == 0:
				continue
			for row in group:
				#print label
				label_count=label_count+1
			#print label_count
			proportion = label_count / float(size_lr)
			add=(proportion * (1.0 - proportion))
			gini =gini + add
	return gini

				
def get_split(X,Y,n_features):
	#print Y
	root_index=float("inf")
	root_value=float("inf")
	root_gini=float("inf")
	
	label_set = list(set(row for row in Y))
	#print label_set
	features=list()
	#features store the indexes of features
	while len(features) < n_features:
		#print len(X[0])
		index = random.randint(0, len(X[0])-1)
		if index not in features:
			features.append(index) # features is storing random feature number out of 28
		
	for index in features:#going through the indexes of all features #row[index]=value of feature 
		for row in X:#for each row in x we get a left and right array

			#print "index1" , index , row[index] # row[index] is the ith element of the row
			groups=tree_split_right_left(index,row[index],X,Y)
				#print Y
			gini = gini_index(groups, label_set)
			if gini<root_gini:
				root_index=index
				root_value=row[index]
				root_gini=gini
				root_groups=groups

	#print root_index,root_value
	return {'index': root_index, 'value': root_value, 'groups': root_groups}

def point_to_terminal(group): #for terminal nodes 
	#for row in group:
	#	print row[1]
	outcomes = list(set(row[1] for row in group))
	outcome_set=set(outcomes)
	#print outcome_set
	return max(outcome_set, key=outcomes.count)


def split(node, max_depth, min_size, n_features, depth):
	left, right = node['groups']
	del(node['groups'])
	if not left or not right:
		node['left'] = node['right'] = point_to_terminal(left + right)
		return
	if depth >= max_depth:
		node['left'], node['right'] = point_to_terminal(left), point_to_terminal(right)
		return
	if len(left) <= min_size:
		node['left'] = point_to_terminal(left)
	else:
		y_left=[]
		x_left=[]

		for i in range(0,len(left)):
			x_left.append(left[i][0])
			y_left.append(left[i][1])
		#print x_left
		node['left'] = get_split(x_left,y_left, n_features)
		#node['left'] = get_split(left, n_features)
		split(node['left'], max_depth, min_size, n_features, depth+1)
    # process right child
	if len(right) <= min_size:
		node['right'] = point_to_terminal(right)
	else:
		x_right=[]
		y_right=[]
		for i in range(0,len(right)):
			x_right.append(right[i][0])
			y_right.append(right[i][1])
		node['right'] = get_split(x_right,y_right,n_features)
		#node['right'] = get_split(right, n_features)
		split(node['right'], max_depth, min_size, n_features, depth+1)
 
##############################################################################################
		#print X
def sample_split_X_Y(sample):
	sample_X=list()
	sample_Y=list()
	for i in range(0,len(sample)):
		sample_X.append(sample[i][0])
		sample_Y.append(sample[i][1])
	#print sample_X
	return sample_X,sample_Y	

def RandomForest(X,Y,X_test,Y_test,folds,max_depth,min_size,sample_size,n_trees,n_features):
	#print len(X)
	trees = list()
	for i in range(n_trees):
		sample=list()

		n_sample = round(len(X) * sample_size)#no . of samples taken from the dataset
		while len(sample) < n_sample:
			index = random.randint(0, len(X)-1)
			#print index
			sample.append([X[index],Y[index]])#sample has random values of random indexed samples stores from X

    	#print sample[0][0]

    	X,Y=sample_split_X_Y(sample)
################BUILD TREE######################################################
    	root=get_split(X,Y,n_features)
    	split(root, max_depth, min_size, n_features, 1)
    	trees.append(root)
	return trees

def predict(depth,tree,row):
	
	if row[tree['index']] < tree['value']:
			#print type(tree['left'])
		#print depth
		if type(tree['left'])==types.DictType:
			return predict(depth,tree['left'],row)
		else:

			return tree['left']
			depth=depth+1
	else:
		#print depth
		if type(tree['right'])==types.DictType:
			return predict(depth,tree['right'],row)

		else:
			return tree['right']
		depth=depth+1
def bagging_predict(trees, row):
	predictions=list()
	for tree in trees:
		predictions.append(predict(depth,tree,row))
	return max(set(predictions), key=predictions.count)
 

def calculate_accuracy(Y_pred,Y_test):
	tp=0
	tn=0
	fn=0
	fp=0
	all_pos=0
	for i in range(0,len(Y_pred)):
		#print "Y_test[i]" ,Y_test[i]
		#print "Y_pred[i]" , Y_pred[i]
		if Y_test[i]==Y_pred[i]:
			tp=tp+1
	return 100 * tp/float(len(Y_pred))

if __name__=="__main__":

	X, y = loadData(sys.argv[1])
	X=sklearn.preprocessing.scale(X)
	#print X
	feel_dict = {"neutral":1,"disgust":1,"panic":0,"anxiety":0,"hot":0,"cold":1,"despair":-1,"sadness":1,"elation":0,"happy":1,"interest":1,"boredom":-1,"shame":1,"pride":1,"contempt":1}
	Y = []
	for item in y:
		Y.append(feel_dict[item])
	#print Y
	folds=5
	max_depth = 30
	min_size = 1
	sample_size = 1.0 # ratio of the data of the entire dataset taken to build trees
	n_trees=10
	#n_features = int(math.sqrt(len(X[0])))
	n_features = .5*len(X[0])

	for i, n_trees in enumerate((1,5,10,15,20,25,30,35,40,45,50)):
		print "******************************************"
		sum=0
		for j, max_depth in enumerate((5,10,15,20,25,30,35,40)):
			list_indexes=split_in_folds(X,Y,folds)
		#print list_indexes

			for item in list_indexes:
				x_train = []
				x_test = []
				y_train = []
				y_test = []
				train_in = item[0]
				test_in = item[1]
				for i in train_in:
					x_train.append(X[i])
					y_train.append(Y[i])
				for i in test_in:
					x_test.append(X[i])
					y_test.append(Y[i])

			#seed(2)
			trees=RandomForest(x_train,y_train,x_test,y_test,folds,max_depth,min_size,sample_size,n_trees,n_features)
			#######TREES NOT PRINTING 
			#print trees
			depth=0
			predictions=list()
			for row in x_test:
				predictions.append(bagging_predict(trees,row))
			Y_pred=predictions

			accuracy=calculate_accuracy(Y_pred,y_test)
			print "no.of trees ",n_trees,"max_depth" , max_depth,"accuracy" , accuracy
			sum=sum+accuracy
		MeanAccuracy= float(sum/(j+1))
		print "MeanAccuracy", MeanAccuracy




