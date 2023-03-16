import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

img = cv2.imread('data/bird.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 1-1.1-2
def km_clust(array, n_clusters):
    # Define the k-means clustering problem
    k_m = KMeans(n_clusters = n_clusters)
    # Solve the k-means clustering problem
    k_m.fit(array)
    # Get the coordinates of the clusters centres
    center = k_m.cluster_centers_
    # Get the label of each point
    label = k_m.labels_
    return(label, center)

h, w, c = img.shape
# Added location feature
Z = np.zeros((h, w, c + 2))
for h in range(len(img)):
    for w in range(len(img[h])):
        Z[h][w] = np.append(img[h][w], [[h], [w]])
Z = Z.reshape(-1, 5)
Ks = [2, 4, 8, 16, 32]
plt.figure(figsize=(16, 12))

for i, K in enumerate(Ks):
    label, center = km_clust(Z, K)

    # Now convert back into uint8, and make original image
    center = np.uint8(center[:, :3])
    res = center[label.flatten()]
    res = res.reshape((img.shape))

    plt.subplot(1, 5, i + 1)
    plt.axis('off')
    plt.title("K = {}".format(K))
    plt.imshow(res)

plt.show()

# 1-3
h, w, c = img.shape
# Added location feature
Z = np.zeros((h, w, c + 2))
for h in range(len(img)):
    for w in range(len(img[h])):
        Z[h][w] = np.append(img[h][w], [[h/4], [w/4]])
Z = Z.reshape(-1, 5)
Ks = [2,4,8,16,32]
plt.figure(figsize=(16, 12))

for i, K in enumerate(Ks):
    label, center = km_clust(Z, K)

    # Now convert back into uint8, and make original image
    center = np.uint8(center[:, :3])
    res = center[label.flatten()]
    res = res.reshape((img.shape))

    plt.subplot(1, 5, i + 1)
    plt.axis('off')
    plt.title("K = {}".format(K))
    plt.imshow(res)

plt.show()
