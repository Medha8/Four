from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
import ipdb
num_features = 14
feel_dict = {"neutral":0,"disgust":-1,"panic":-2,"anxiety":-3,"hot":-4,"cold":-5,"despair":-6,"sadness":-7,"elation":1,"happy":2,"interest":3,"boredom":4,"shame":5,"pride":6,"contempt":7}
data = np.genfromtxt("./test_upd.csv",delimiter=',',dtype=np.float, converters={0:lambda x: np.float(feel_dict[x])})
print data
test_data = data[0::5,:]
indices = np.ones(data.shape, dtype=bool)
indices[0::5,:] = 0
train_data = data[indices]
train_data = train_data.reshape((len(train_data)/num_features,num_features))

X_train_data = train_data[:,1:]
Y_train_data = train_data[:,0]
X_test_data = test_data[:,1:]
Y_test_data = test_data[:,0]



features = data[:,1:]
labels = data[:,0]
model = KMeans(n_clusters=1, random_state=0).fit(features)
features = np.asarray(features)
Y_train_predict = model.predict(features)

training_accuracy = accuracy_score(Y_train_predict, labels)

print "clusters","training","testing"
for i in range(1,15):
	model = KMeans(n_clusters=i, random_state=0).fit(X_train_data)
	Y_train_predict = model.predict(X_train_data)
	training_accuracy = accuracy_score(Y_train_data, Y_train_predict, normalize=True, sample_weight=None)
	Y_test_predict = model.predict(X_test_data)
	testing_accuracy = accuracy_score(Y_test_data,Y_test_predict,normalize=True, sample_weight=None)
	print i, training_accuracy, testing_accuracy

#kmeans = KMeans(n_clusters=2, random_state=0).fit(r)
# for k in labels:
# 	label_numbers[k] = dictionary[labels[k]]
# print label_numbers
# kmeans.labels_array(label_numbers, dtype=int32)