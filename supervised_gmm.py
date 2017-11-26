import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.stats import multivariate_normal as normal
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit
import statsmodels
import sklearn.datasets as datasets
 
def loadData(filename):
	with open(filename) as f:
		content = f.readlines()
	content = [x.split() for x in content]
	label = []
	feature_vec = []
	for temp in content:
		label.append(temp[0])
		a = np.asarray(temp[1:])
		x = []
		for t in a:
			x.append(float(str(np.round(float(t),2))))
		feature_vec.append(np.asarray(x))
	return label, feature_vec

def condn_probab(X , mean_mat, cov):
	PSD_cov = np.outer(cov, cov)
	#PSD_cov = statsmodels.stats.correlation_tools.cov_nearest(cov)
	#print PSD_cov
	k = normal.pdf(X , mean_mat, cov, allow_singular=True)
	return k 

if __name__=="__main__":

	y, feature_vec = loadData(sys.argv[1])
 	#feel_dict = {"neutral":0,"disgust":-1,"panic":-2,"anxiety":-0,"hot":-4,"cold":-5,"despair":-6,"sadness":-7,"elation":1,"happy":2,"interest":0,"boredom":4,"shame":5,"pride":6,"contempt":7}
	feel_dict = {
	"anxiety":2,
	"boredom": 2, 
	"cold": 2, 
	"contempt": 0,
	"despair": 2,
	"disgust": 0,
	"elation": 1,
	"happy": 1,
	"hot": 1, 
	"interest": 1,
	"neutral": 2,
	"panic": 0,
	"pride": 2,
	"sadness": 0, 
	"shame": 2
	}

	X_new = np.asarray(preprocessing.scale(feature_vec))

	Y_new = []
	for item in y:
		Y_new.append(feel_dict[item])

	rs = ShuffleSplit(n_splits=5, test_size=.10, random_state=0)
	lis = []
	for train_index, test_index in rs.split(X_new):
		lis.append([train_index, test_index])
	acc = []
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

	#initialise initial clusters with 
		no_clusters = 3
		no_features = 28
		n = len(Y) #no of X

		##initialise means
		p = np.zeros(no_features)
		mean_mat = [p, p, p]
		corr_ind = np.zeros(n)
		num = [0, 0, 0]
		lamda = [0, 0, 0] 

		for i in range(0, no_clusters):
			for j in range(0, len(X)):
				if Y[j] == i:
					num[i] += 1
					mean_mat[i] = mean_mat[i] + np.asarray(X[j])
		#print mean_mat

		for i in range(0,len(mean_mat)):
			mean_mat[i] = [x / num[i] for x in mean_mat[i]]


		##initialise variance 
		
		cov = []
		sum_weight = [0, 0, 0]
		total_weight = 0
		for i in range(0, no_clusters):
			sigma = np.zeros((no_features,no_features))
			for j in range(0, n):
				if Y[j] == i:
					sigma +=np.multiply( np.outer(np.subtract(X[j], mean_mat[i]), np.subtract(X[j], mean_mat[i])), 1)
					sum_weight[i] += 1
			total_weight += sum_weight[i]
			cov.append(sigma)
		cov = np.divide( cov, total_weight)
		#print cov
		'''
		prior = [1, 1, 1]
		for i in range(0, no_clusters):
			data = []
			for j in range(0, len(X)):
				if Y[j] == i:
					data.append(X[j])
			sigma = np.cov(data)
			print sigma.shape
			cov.append(sigma)
		'''
		#cov1 = datasets.make_spd_matrix(28, random_state=None)
		#cov = [cov1, cov1, cov1]

		for i in range(0, len(lamda)):
			lamda[i] = float(num[i])/(n)
		
		while True:
			# expectation
			#print lamda
			p = np.zeros(len(X))
			wci = [p, p, p]

			for i in range(0, n):
				mle = float('inf')
				for j in range(0, no_clusters):
					# "noerror"
					pX_ = np.multiply(condn_probab(np.asarray(X[i]), mean_mat[j], cov[j]), lamda[j])
					wci[j][i] = pX_
					if pX_ < mle:
						mle = pX_
						assign = j
				if assign!= corr_ind[i]:
					corr_ind[i] = assign

			# maximization
			prev_lamda = lamda
			prev_mean_mat = mean_mat
			p = np.zeros(no_features)
			mean_mat = [p, p, p]
			lamda = [0, 0, 0]
			sum_weight = [0, 0, 0]
			total_weight = 0
			
			#----------------------------
			## mean
			for i in range(0, no_clusters):
				for j in range(0, n):
					mean_mat[i] += np.multiply(np.asarray(X[j]), wci[i][j])
					sum_weight[i] += wci[i][j]
				total_weight += sum_weight[i]
			
			for i in range(0, no_clusters):
				mean_mat[i][:] = [x / total_weight for x in mean_mat[i]]
			
			## --------------------------put condition on lambda when to stop
			## lamda
			for i in range(0, no_clusters):
				lamda[i] = float(sum_weight[i])/(total_weight)
			diff = np.subtract(lamda, prev_lamda)
			if LA.norm(diff) < 0.2:
				break
			#------------------------------
			total_weight = 0
			sum_weight = [0, 0, 0]

			## covariance using dan's method
			cov = []
			for i in range(0, no_clusters):
				sigma = np.zeros((no_features,no_features))
				for j in range(0, n):
					if Y[j] == i:
						sigma +=np.multiply( np.outer(np.subtract(X[j], prev_mean_mat[i]), np.subtract(X[j], prev_mean_mat[i])), wci[i][j])
						sum_weight[i] += wci[i][j]
				total_weight += sum_weight[i]
				cov.append(sigma)

			cov = np.divide( cov, total_weight)

		pred_ytest = []
		for i in range(0, len(x_test)):
			mle = float(0)
			for j in range (0, no_clusters):
				pX_ = condn_probab(np.asarray(x_test[i]), prev_mean_mat[j] , cov[j])
				if pX_ > mle:
					mle = pX_
					assign = j
			pred_ytest.append(assign)
		acc.append(accuracy_score(y_test, pred_ytest, normalize=True, sample_weight=None))
	print np.mean(acc)
