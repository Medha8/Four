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
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import csv

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
	
	pca = PCA(n_components=28)
	Xnew = pca.fit_transform(X)
	list_ = []
	for i in range(0,len(Xnew)):
		feature = Xnew[i]
		bk = feature.tolist()
		bk = ' '.join([str(x) for x in bk])
		list_.append(y[i] + ' ' + bk)
	
	thefile = open('final_pca.txt', 'w')
	for item in list_:
		thefile.write("%s\n" % item)

	X_ = pca.components_
	var=pca.explained_variance_ratio_
	#cumulative variance explained 
	var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=2))
	#print var
	print var1

	fig = plt.figure()
	plt.xlabel('ncomponents')
	plt.ylabel('cum_explained_variance_ratio_')
	#plt.plot(pca.explained_variance_ratio_)
	plt.plot(var1)

	plt.show()
	plt.cla()
	plt.clf()
	plt.close(fig)

