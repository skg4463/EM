import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.cluster import KMeans
from sklearn import preprocessing as pre


CONVERGENCE_THRESHOLD = 100


def EM(X, W, H, center, scale):
    plt_num = 1

    estimators = {'k_means_2': KMeans(n_clusters=2),
                  'k_means_3': KMeans(n_clusters=3),
                  'k_means_4': KMeans(n_clusters=4)}

    for name, est in estimators.items():
        print(name)
        J = int(name[8:])
        NUM_PIXELS = X.shape[0]

        # perform k means
        est.fit(X)
        segments_id = est.labels_

        # get initial cluster centers/ means from k-means
        means = est.cluster_centers_

        # get initial pi from k-means
        pi = np.array([np.sum(segments_id == i) for i in range(J)])
        pi = pi / float(NUM_PIXELS)

        print(0 not in pi)

        ### EM ###
        prev_Q = sys.maxsize
        list_of_q = []

        while True:
            # E-Step
            ll = np.zeros((NUM_PIXELS, J))
            for j in range(J):
                ll[:, j] = -0.5 * np.sum((X - means[j,]) ** 2, 1)

            # compute w_ij
            w = np.exp(ll) @ np.diag(pi)
            w = (w.T / np.sum(w, 1)).T

            # compute Q without constant K
            Q = np.sum(ll * w)
            list_of_q.append(Q)

            # check for convergence
            if abs(Q - prev_Q) <= CONVERGENCE_THRESHOLD:
                break
            else:
                prev_Q = Q

            # M-Step

            # update means
            for j in range(J):
                means[j,] = np.sum((X.T * w[:, j]).T, 0) / np.sum(w[:, j])

            # update pi
            pi = np.sum(w, 0) / NUM_PIXELS

        # plot convergence of Q as we progress through EM
        plt.figure(plt_num)
        plt.plot(list_of_q)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Q at E-step")
        plt.show()
        plt_num += 1

        # display result as segmented image
        segmented_img_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for i in range(H):
            for j in range(W):
                idx = (i - 1) * W + j
                pixel = X[idx,]
                pixel_segment_id = np.argmax(w[idx,])
                segmented_img_rgb[i, j,] = means[pixel_segment_id,] * scale + center

        plt.figure(plt_num)
        plt.imshow(segmented_img_rgb)  # show segmented image
        plt.show()
        plt_num += 1


# read data
img = cv2.imread('images/Lena.jpg')
(H, W, N) = img.shape
data = img.reshape((H * W, N))
data_centers = np.mean(data, 0)
data_scale = np.std(data, 0)
data = pre.scale(data)

# run EM
EM(data, W, H, data_centers, data_scale)
