from sklearn.cluster import MeanShift, estimate_bandwidth


def meanshift(X, bandwidth):
    if bandwidth is None:
        bandwidth = estimate_bandwidth(X)

    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    return cluster_centers, labels
