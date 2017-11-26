import sys
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC

def loadData(filename):
	X = []
	Y = []
	with open(filename) as f:
		content = f.readlines()
	for item in content:
		X.append(item.split()[1:])
		Y.append(item.split()[0])
	return X, Y

if __name__=="__main__":
	X_, y = loadData(sys.argv[1])
	X = preprocessing.scale(X_)

	num_features = 7
	feel_dict = {"neutral":6,
	"disgust":2,
	"panic":0,
	"anxiety":0,
	"hot":0,
	"cold":2,
	"despair":4,
	"sadness":2,
	"elation":1,
	"happy":3,
	"interest":3,
	"boredom":5,
	"shame":2,
	"pride":3,
	"contempt":2}

	Y = []
	for item in y:
		Y.append(feel_dict[item])
	#print Y
	for classifier in [ 'rbf', 'poly']:
		for i,C in enumerate((.0001, .001, .01, .1,1,10,100,1000)):
			rs = ShuffleSplit(n_splits=50, test_size=.25, random_state=0)
			lis = []
			for train_index, test_index in rs.split(X):
				lis.append([train_index, test_index])

			#print lis
			train = []
			test = []
			for item in lis:
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
				
				#
				clf = svm.SVC(kernel=classifier,gamma="auto" , C=C)
				clf.fit(x_train, y_train) 
				'''
				clf = LinearSVC(penalty='l2', random_state=0)
				clf.fit(x_train, y_train)
				'''
				ytrain_pred=clf.predict(x_train)
				ytest_pred=clf.predict(x_test)

				train_accuracy = accuracy_score(ytrain_pred, y_train)
				test_accuracy= accuracy_score(ytest_pred,y_test)
				train.append(train_accuracy)
				test.append(test_accuracy)
				tr_error = mean_squared_error(y_train, ytrain_pred, sample_weight=None)
				test_error = mean_squared_error(y_test, ytest_pred, sample_weight=None)
				#print "train_accuracy = ", train_accuracy, " test_accuracy = ", test_accuracy
			#print "train_error = ", tr_error, " test_error = ", test_error
			print "classifier", classifier ,"value of C=", C, " train=", np.mean(train), " test=", np.mean(test)



#kmeans = KMeans(n_clusters=2, random_state=0).fit(r)
# for k in labels:
# 	label_numbers[k] = dictionary[labels[k]]
# print label_numbers
# kmeans.labels_array(label_numbers, dtype=int32)
