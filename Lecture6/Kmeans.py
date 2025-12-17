#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 15:07:00 2025

@author: mahmutbagci
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris



iris = load_iris()
iris = sns.load_dataset('iris')
iris.head()


X = iris.iloc[:, :-2]
#y = iris['target']
y = iris.species

wcss = []
#ELBOW method

#n_init = By default is 10 and so the algorithm will initialize 
#the centroids 10 times and will pick the most converging value as the best fit.
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,
                    init='k-means++',
                    max_iter=300,
                    n_init=10,
                    random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
# from above array with help of elbow method
#we can get no of cluster to provide.
kmeans = KMeans(n_clusters=3,
                init='k-means++',
                max_iter=300,
                n_init=10,
                random_state=0)
y_kmeans = kmeans.fit_predict(X)


# Visualising the clusters
cols = iris.columns
plt.scatter(X.loc[y_kmeans == 0, cols[0]],
            X.loc[y_kmeans == 0, cols[1]],
            s=100, c='purple',
            label='Iris-setosa')
plt.scatter(X.loc[y_kmeans == 1, cols[0]],
            X.loc[y_kmeans == 1, cols[1]],
            s=100, c='orange',
            label='Iris-versicolour')
plt.scatter(X.loc[y_kmeans == 2, cols[0]],
            X.loc[y_kmeans == 2, cols[1]],
            s=100, c='green',
            label='Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=100, c='red',
            label='Centroids')

plt.legend()

pd.crosstab(y, y_kmeans)