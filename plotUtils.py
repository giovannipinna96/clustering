import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import silhouette_samples
import numpy as np


def plot_blob(X, c='white', edgecolor='black', s=50):
    plt.scatter(X[:, 0], X[:, 1], c=c, marker='o', edgecolor=edgecolor, s=s)
    plt.grid()
    plt.tight_layout()


def plot_three_clusters(X, y_km):
    plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], c='lightgreen', marker='s', edgecolor='black', s=50, label='Cluster1')
    plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], c='orange', marker='o', edgecolor='black', s=50, label='Cluster2')
    plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], c='lightblue', marker='v', edgecolor='black', s=50, label='Cluster3')
    plt.grid()
    plt.tight_layout()


def plot_two_clusters(X, y):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='lightgreen', marker='s', edgecolor='black', s=50, label='Cluster1')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='orange', marker='o', edgecolor='black', s=50, label='Cluster2')
    plt.grid()
    plt.tight_layout()


def plot_silhouette(X, y_km):
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                 edgecolor='none', color=color)

        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--")
    plt.yticks(yticks, cluster_labels + 1)
