import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from Data import create_moons
from plotUtils import plot_two_clusters

X, _ = create_moons()

db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_db = db.fit_predict(X)

plot_two_clusters(X, y_db)
plt.legend()
plt.title('DBSCAN')
plt.tight_layout()
plt.show()
