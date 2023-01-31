import numpy as np

def kmeans(X:np.ndarray, k:int, centroids=None, max_iter=30, tolerance=1e-2):

    # Select k unique points from X as starting centroids
    num_rows = X.shape[0]
    if centroids == 'kmeans++': 
        centroids = select_centroids(X,k)
    else:
        centroids = X[np.random.randint(num_rows, size=k), :]
    labels = np.ones(num_rows)

    # Until max iterations are reached
    for num in range(num_rows):

        # Find best labels for the new centroids
        for i, record in enumerate(X):

            # Find distances to each current centroid for this record
            distances = np.linalg.norm(record-centroids, axis=1)

            # Assign the record to the min distance centroid
            labels[i] = np.argmin(distances)

        # Calculate new centroids
        new_centroids = np.copy(centroids)
        for j, centroid in enumerate(new_centroids):
            new_centroids[j] = np.mean(X[np.where(labels == j)], axis=0)

        # Stop if centroids aren't changing
        diff = np.average(np.linalg.norm(new_centroids-centroids, axis=1))
        if diff < tolerance:
            break
        centroids = new_centroids

    return centroids, labels.astype(int)


def select_centroids(X,k):
    """
    kmeans++ algorithm to select initial points:

    1. Pick first point randomly
    2. Pick next k-1 points by selecting points that maximize the minimum
       distance to all existing clusters. So for each point, compute distance
       to each cluster and find that minimum.  Among the min distances to a cluster
       for each point, find the max distance. The associated point is the new centroid.

    Return centroids as k x p array of points from X.
    """

    # Radomly pick the first point
    centroids = X[np.random.randint(X.shape[0], size=1), :]

    for i in range(k-1):
        min_distances = np.zeros(X.shape[0])
        for index, record in enumerate(X):
            min_distances[index] = np.linalg.norm(record-centroids, axis=1).min()

        centroids = np.vstack((centroids, X[np.argmax(min_distances)]))

    return centroids