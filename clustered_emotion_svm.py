import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import pandas as pd


plt.close('all')
count_length=0
X = []; Y = []
with open("test.csv", 'r') as f:
	#line = file.readline()
	for line in f:
		if line.strip():
			line = line.strip().split(',')
			
 		
 		#print line
 			#for u in line:
 			#	u=float(u)
 			#print line[12]
			Y.append(line[0])
			X.append([float(x) for x in line[1:]])
			count_length=count_length+1
			
print count_length
 			#count=count+1
#print Y[1]	
X = np.asarray(X); Y = np.asarray(Y)

for i in range(count_length):

	if Y[i]=="hot" or Y[i]=="panic":
		print Y[i]
		Y[i]=1
		print "detected hot"
		print Y[i]
	elif Y[i]=="elation":
		print Y[i]

		Y[i]=3
		print Y[i]

	elif Y[i]=="neutral":
		print Y[i]

		Y[i]=5
		print Y[i]

	elif Y[i]=="pride" or Y[i]=="interest" or Y[i]=="happy":
		print Y[i]

		Y[i]=6
		print Y[i]

	elif Y[i]=="despair":
		print Y[i]

		Y[i]=7
		print Y[i]

	elif Y[i]=="boredom":
		print Y[i]

		Y[i]=8
		print Y[i]

	else:
		print Y[i]

		Y[i]=4
		print Y[i]


X = StandardScaler().fit_transform(X)
#print X


#for i,C in enumerate((.1,1,10,100,1000,10000,100000,1000000)):
clf = svm.SVC(kernel='rbf',gamma=5 , C=1)
clf.fit(X, Y) 


Y_pred=clf.predict(X)
	#print Y_pred
train_accuracy = accuracy_score(Y_pred, Y)
print i,train_accuracy


fig = plt.figure()
#plt.plot(pca.explained_variance_ratio_)

plot_decision_regions(X, Y, classifier=clf)
#plt.legend(loc='upper left')
plt.tight_layout()
plt.show()