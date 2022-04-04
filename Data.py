from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def create_blob(n_sample=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=1):
    X, y = make_blobs(n_samples=n_sample, n_features=n_features, centers=centers, cluster_std=cluster_std,
                      shuffle=shuffle, random_state=random_state)
    return X, y


def create_moons(n_sample=150, noise=0.05, random_state=1):
    X, y = make_moons(n_samples=n_sample, noise=noise, random_state=random_state)
    return X, y


def get_standardize_data(X_train, X_test=None):
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    if X_test != None:
        X_test_std = sc.transform(X_test)
        return X_train_std, X_test_std
    return X_train_std


def get_split_data(X, y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size, random_state=1)
    return X_train, X_test, y_train, y_test


def get_split_std_data(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = get_split_data(X, y, test_size, random_state)
    X_train_std, X_test_std = get_standardize_data(X_train, X_test)
    return X_train_std, X_test_std, y_train, y_test
