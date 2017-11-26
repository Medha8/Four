import sys
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn
from sklearn import mixture


def make_ellipses(gmm, ax):
	for n, color in enumerate('rgb'):
		v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
		u = w[0] / np.linalg.norm(w[0])
		angle = np.arctan2(u[1], u[0])
		angle = 180 * angle / np.pi  # convert to degrees
		v *= 9
		ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
			180 + angle, color=color)
		ell.set_clip_box(ax.bbox)
		ell.set_alpha(0.5)
		ax.add_artist(ell)

def load(filename):
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

if __name__=="__main__":
	label, feature_vec = load(sys.argv[1])
 	print "test"
 	#feel_dict = {"neutral":0,"disgust":-1,"panic":-2,"anxiety":-3,"hot":-4,"cold":-5,"despair":-6,"sadness":-7,"elation":1,"happy":2,"interest":3,"boredom":4,"shame":5,"pride":6,"contempt":7}
	feel_dict = {"neutral":0,"disgust":-1,"panic":-1,"anxiety":-1,"hot":-1,"cold":-1,"despair":-1,"sadness":-1,"elation":1,"happy":1,"interest":1,"boredom":1,"shame":1,"pride":1,"contempt":1}
	feature_vec = np.asarray(feature_vec)
	val = []
	for temp in label:
		val.append(feel_dict[temp])
	val = np.asarray(val)

	skf = StratifiedKFold(n_splits=2, shuffle = True)
	for train_index, test_index in skf.split(feature_vec, val):
		X_train, X_test = feature_vec[train_index], feature_vec[test_index]
		y_train, y_test = val[train_index], val[test_index]
	

	n_classes = len(np.unique(y_train))
	classifiers = dict((covar_type, mixture.GMM(n_components=n_classes,
                    covariance_type=covar_type, init_params='wc', n_iter=20))
                   for covar_type in ['spherical', 'diag', 'tied', 'full'])	
	n_classifiers = len(classifiers)
	for index, (name, classifier) in enumerate(classifiers.items()):
		print "##############"
		print X_train.shape
		classifier.fit(X_train)		
		y_train_pred = classifier.predict(X_train)
		tr = sklearn.metrics.accuracy_score(y_train, y_train_pred, normalize=True, sample_weight=None)
		y_test_pred = classifier.predict(X_test)
		tes =sklearn.metrics.accuracy_score(y_test, y_test_pred, normalize=True, sample_weight=None)
		print name, ":name", tr ,":train"
		print name, ":name",tes, ":tes"



