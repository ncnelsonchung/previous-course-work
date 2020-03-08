from __future__ import print_function
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics

def create_data():
    # Generate sample points
    centers = [[3,5], [5,1], [8,2], [6,8], [9,7]]
    X, y = make_blobs(n_samples=1000,centers=centers,cluster_std=0.5,random_state=3320)
    # =======================================
    # Complete the code here.
    # Plot the data points in a scatter plot.
    # Use color to represents the clusters.
    # =======================================
    # colors = ("red", "green", "blue", "orange", "purple", "yellow")
    # plt.scatter(X[:, 0], X[:, 1], label=y, c=list(map(lambda x: colors[x], y)))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm.get_cmap('RdBu'))
    plt.show()
    return [X, y]

def my_clustering(X, y, n_clusters):
    # =======================================
    # Complete the code here.
    # return scores like this: return [score, score, score, score]
    # =======================================
    clustering = KMeans(n_clusters=n_clusters)
    clustering.fit(X)
    # colors = ("red", "green", "blue", "orange", "purple", "yellow")
    # plt.scatter(X[:, 0], X[:, 1], label=clustering.labels_, c=list(map(lambda x: colors[x], clustering.labels_)))
    plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_, cmap=cm.get_cmap('RdBu'))
    centers = np.array(clustering.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], marker="x", c="black")
    plt.show()
    labels_true = y
    labels_pred = clustering.labels_
    ari_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    mri_score = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    v_measure_score = metrics.v_measure_score(labels_true, labels_pred)
    silhouette_avg = metrics.silhouette_score(X, labels_pred)
    return [ari_score, mri_score, v_measure_score, silhouette_avg]

def main():
    X, y = create_data()
    range_n_clusters = [2, 3, 4, 5, 6]
    ari_score = [None] * len(range_n_clusters)
    mri_score = [None] * len(range_n_clusters)
    v_measure_score = [None] * len(range_n_clusters)
    silhouette_avg = [None] * len(range_n_clusters)

    for n_clusters in range_n_clusters:
        i = n_clusters - range_n_clusters[0]
        print("Number of clusters is: ", n_clusters)
        [ari_score[i], mri_score[i], v_measure_score[i], silhouette_avg[i]] = my_clustering(X, y, n_clusters)
        print('The ARI score is: ', ari_score[i])
        print('The MRI score is: ', mri_score[i])
        print('The v-measure score is: ', v_measure_score[i])
        print('The average silhouette score is: ', silhouette_avg[i])

    # =======================================
    # Complete the code here.
    # Plot scores of all four evaluation metrics as functions of n_clusters.
    # =======================================

    import pandas as pd
    df = pd.DataFrame([ari_score, mri_score, v_measure_score, silhouette_avg],
                      index=["ari_score", "mri_score", "v_measure_score", "silhouette_avg"], columns=range_n_clusters)
    df = df.T
    # print(df)
    df.plot(kind="bar")
    plt.xlabel("no. of cluster")
    plt.show()

if __name__ == '__main__':
    main()

