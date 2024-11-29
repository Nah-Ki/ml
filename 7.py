import pylab as pl
import numpy as np
from sklearn.cluster import KMeans

np.random.seed(110)

red_mean = 3
red_std = 0.8
blue_mean = 7
blue_std = 1

red = np.random.normal(red_mean, red_std, size=40)
blue = np.random.normal(blue_mean, blue_std, size=40)
both_colours = np.sort(np.concatenate((red, blue)))
y = np.zeros(len(both_colours))

kmeans = KMeans(n_clusters=2)
kmeansoutput = kmeans.fit(both_colours.reshape(-1, 1))

Nc = range(1, 5)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(both_colours.reshape(-1, 1)).score(both_colours.reshape(-1, 1)) for i in range(len(kmeans))]

pl.plot(Nc, score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()

pl.scatter(both_colours, y, c=kmeansoutput.labels_)
pl.xlabel('Data points')
pl.ylabel('None')
pl.title('2 Cluster K-Means')
pl.show()
