import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8, random_state=101) #create blob like fake data that has the specific features that are needed
#creates a 200,2 np array (2 = number of features)
# data[0] is a 200,2 array.
# plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='gist_rainbow')
# data[1] will be the actual labels of the dataset

kmeans = KMeans(n_clusters=4) # we know that we have 4 blobs
kmeans.fit(data[0])

print(kmeans.cluster_centers_) #prints the cluster centers
print(kmeans.labels_) #labels it predicted

#plotting the actual labels and predicted labels

fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)

ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='gist_rainbow')

ax2.set_title('Original')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='gist_rainbow')
plt.show()