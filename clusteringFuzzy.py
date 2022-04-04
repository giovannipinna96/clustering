import matplotlib.pyplot as plt
from fcmeans import FCM
from Data import create_blob
from plotUtils import plot_two_clusters, plot_blob, plot_three_clusters
import numpy as np

X, _ = create_blob()
plot_blob(X)
plt.tight_layout()
plt.show()

fcm = FCM(n_clusters=3)
fcm.fit(X)

centers = fcm.centers
# print(centers)
y = fcm.predict(X)
x1 = np.zeros((1, 2))
x1[0, 0] = np.mean(centers[:, 0]) + 2
x1[0, 1] = np.mean(centers[:, 1]) + 2
y1 = fcm.soft_predict(x1)
print(
    f"x1: {x1}, \n" + f"prob_cl1: {y1[0][0]},  prob_cl2: {y1[0][1]},  prob_cl3: {y1[0][2]}")
plot_three_clusters(X, y)
plt.scatter(x1[0, 0], x1[0, 1], c='red', marker='*')
plt.title('Fuzzy clustering')
plt.legend()
plt.tight_layout()
plt.show()
