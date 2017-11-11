from sklearn.cluster import KMeans
import numpy as np
#import ipdb
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

num_features = 14
feel_dict = {"neutral":0,"disgust":-1,"panic":-2,"anxiety":-3,"hot":-4,"cold":-5,"despair":-6,"sadness":-7,"elation":1,"happy":2,"interest":3,"boredom":4,"shame":5,"pride":6,"contempt":7}
data = np.genfromtxt("test.csv",delimiter=',',dtype=np.float, converters={0:lambda x: np.float(feel_dict[x])})
test_data = data[0::5,:]
indices = np.ones(data.shape, dtype=bool)
indices[0::5,:] = 0
train_data = data[indices]
train_data = train_data.reshape((len(train_data)/num_features,num_features))

X_train_data = train_data[:,1:]
Y_train_data = train_data[:,0]
#print Y_train_data
X_test_data = test_data[:,1:]
Y_test_data = test_data[:,0]

for i,C in enumerate((.1,1,10,100,1000,10000,100000,1000000)):
	clf = svm.SVC(kernel='rbf',gamma=.1 , C=C)
	clf.fit(X_train_data, Y_train_data) 


	Y_pred_train=clf.predict(X_train_data)
	Y_pred_test=clf.predict(X_test_data)

	print Y_pred
	train_accuracy = accuracy_score(Y_pred_train, Y_train_data)
	test_accuracy= accuracy_score(Y_pred_test,Y_test_data)
	print i,test_accuracy



#kmeans = KMeans(n_clusters=2, random_state=0).fit(r)
# for k in labels:
# 	label_numbers[k] = dictionary[labels[k]]
# print label_numbers
# kmeans.labels_array(label_numbers, dtype=int32)