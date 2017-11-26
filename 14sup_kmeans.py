import sys
from sklearn.cluster import KMeans
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error


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
	X_new = preprocessing.scale(X_)

	num_features = 14
	feel_dict = {"neutral":1,
	"disgust":2,
	"panic":3,
	"anxiety":4,
	"hot":5,
	"cold":6,
	"despair":7,
	"sadness":8,
	"elation":9,
	"happy":10,
	"interest":11,
	"boredom":12,
	"shame":13,
	"pride":14,
	"contempt":0}

	Y_new = []
	for item in y:
		Y_new.append(feel_dict[item])

	rs = ShuffleSplit(n_splits=5, test_size=.10, random_state=0)
	lis = []
	for train_index, test_index in rs.split(X_new):
		lis.append([train_index, test_index])

	for item in lis:
		X = []
		x_test = []
		Y = []
		y_test = []
		train_in = item[0]
		test_in = item[1]
		for i in train_in:
			X.append(X_new[i])
			Y.append(Y_new[i])
		for i in test_in:
			x_test.append(X_new[i])
			y_test.append(Y_new[i])

		p = np.zeros(28)
		#cluster_mean_list = [[] for i in range(num_features)]
		cluster_mean_list = [p, p, p, p, p, p, p, p, p, p, p, p, p, p]
		count_list = np.zeros(num_features)
		for i in range(0,num_features):
			for j in range(0,len(Y)):
				if Y[j] == i:
					count_list[i] = count_list[i] + 1
					cluster_mean_list[i] = cluster_mean_list[i] + np.asarray(X[j])
						
		for i in range(0,len(cluster_mean_list)):
			#print count_list[i]
			cluster_mean_list[i][:] = [x / count_list[i] for x in cluster_mean_list[i]]
		
		no_change = False
		cluster_assign = np.zeros(len(Y))
		iteration = 0
		while no_change == False:
			# recluster
			iteration = iteration + 1
			no_change = True
			for i in range(0, len(X)):
				feature = np.asarray(X[i])
				dis = float('inf')
				for j in range(0,len(cluster_mean_list)):
					cluster_dist = scipy.spatial.distance.euclidean(np.asarray(cluster_mean_list[j]), feature)
					#print j, "cluster == >", cluster_dist
					if cluster_dist < dis:
						assign = j
						dis = cluster_dist
				if iteration == 1:
					cluster_assign[i] = assign
					no_change = False
				else:
					if assign != cluster_assign[i] :
						cluster_assign[i] = assign
						no_change = False
			#print cluster_assign
			#recalculate mean
			if no_change == False:
				p = np.zeros(28)
				cluster_mean_list = [p, p, p, p, p, p, p, p, p, p, p, p, p, p]
				count_list = np.zeros(num_features)
				for i in range(0,num_features):
					for j in range(0,len(cluster_assign)):
						if cluster_assign[j] == i:
							count_list[i] = count_list[i] + 1
							cluster_mean_list[i] = cluster_mean_list[i] + np.asarray(X[j])

				for i in range(0,len(cluster_mean_list)):
					cluster_mean_list[i][:] = [x / count_list[i] for x in cluster_mean_list[i]]
			


		pred = []
		for i in range(0,len(x_test)):
			feature = x_test[i]
			dis = float('inf')
			for j in range(0,len(cluster_mean_list)):
				cluster_dist = scipy.spatial.distance.euclidean(np.asarray(cluster_mean_list[j]), feature)
				if cluster_dist < dis:
					assign = j
					dis = cluster_dist
			pred.append(assign)
		pred = np.asarray(pred)
		count = 0
		missclass = 0
		for i in range(0,len(pred)):
			count = count + 1
			if pred[i] != y_test[i]:
				missclass = missclass + 1
		print float(missclass)/(count)


			
			


		



