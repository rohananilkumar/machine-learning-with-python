import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8, random_state=101) #create blob like fake data that has the specific features that are needed
#creates a 200,2 np array (2 = number of features)
# data[0] is a 200,2 array. Now as center =4 we'll have 4 such np arrays.. hence 4 blobs
plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='gist_rainbow')

plt.show()