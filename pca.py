import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale


#plt.cla()
#plt.clf()
plt.close('all')
count=0
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
			

 			#count=count+1
 		
X = np.asarray(X); Y = np.asarray(Y)
#print X
#Standardize features by removing the mean and scaling to unit variance.Centering and scaling
# happen independently on each feature by computing the relevant statistics on the samples in the training set
X=scale(X)
#X = StandardScaler().fit_transform(X)
print X
pca = PCA(n_components=12)
pca = pca.fit(X)
#The amount of variance that each PC explains
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
#plt.plot(var_values)

