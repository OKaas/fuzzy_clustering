import matplotlib.pyplot as plt  # matlab plot
from mpl_toolkits.mplot3d import Axes3D  # necessary for 3D projection
from matplotlib.widgets import TextBox

import numpy as np  # best of the best
from numpy import linalg as LA
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from sklearn import metrics

# Clustering algorithms
from main.clustering.mean_shift import meanshift

INDEX_X = 0
INDEX_Y = 1
INDEX_Z = 2
INDEX_MEM = 3

MIN_X = -1
MAX_X = 1

MIN_Y = -1
MAX_Y = 1

MAX_GXX = 0.5
MAX_GXY = 0.5

MAX_GYX = 0.5
MAX_GYY = 0.5

STEP_X = 0.1
STEP_Y = 0.1

THRESHOLD_SUBTRACTION = 0.5

NUM_X = 20
NUM_Y = 20

NUMBER_OF_GAUSSIAN = 5

FIGURE_CLUSTERS = None


class Membership:
    def __init__(self):
        self.cluster = []
        self.membership = []

    def add_influence(self, p_cl, p_membership):
        self.cluster.append(p_cl)
        self.membership.append(abs(p_membership))

    def sum_membership(self):
        return np.sum(self.membership)

    @property
    def membership(self):
        return self.__membership

    @membership.setter
    def membership(self, value):
        self.__membership = value

    @property
    def cluster(self):
        return self.__cluster

    @cluster.setter
    def cluster(self, value):
        self.__cluster = value


def arrange_function(p_x, p_y, p_influence, p_cluster, p_data):
    p_data[INDEX_Z][p_x][p_y] += p_influence
    p_data[INDEX_MEM][p_x][p_y].add_influence(p_cluster, p_influence)


def arrange_gaussian(p_x, p_y, p_gxx, p_gxy, p_gyx, p_gyy, p_cl, substract, p_data):
    rv = multivariate_normal([p_x, p_y], [[p_gxx, p_gxy], [p_gyx, p_gyy]])

    add = rv.pdf(pos)

    if substract:
        p_data[INDEX_Z] -= add
    else:
        p_data[INDEX_Z] += add

    for idx in range(0, add[0].size):
        for idy in range(0, add[1].size):
            p_data[INDEX_MEM][idx][idy].add_influence(p_cl, add[idx][idy])


def calculate_clustering(data):
    print(data)


def distance(p_a, p_b):
    return LA.norm(p_a - p_b)


# def arrange_function(p_data):
#     xxx = p_data[INDEX_X]
#     yyy = p_data[INDEX_Y]
#
#     return xxx * (1 - xxx) * np.cos(4 * np.pi * xxx) * np.sin(4 * np.pi * yyy ** 2) ** 2


if __name__ == '__main__':
    x = np.linspace(MIN_X, MAX_X, NUM_X, endpoint=True)
    y = np.linspace(MIN_Y, MAX_Y, NUM_Y, endpoint=True)

    # create matrix for data
    xx, yy = np.meshgrid(x, y)

    # Z value is not know yet -> will be changed by functions
    zz = xx.copy()
    zz.fill(0)
    pos = np.dstack((xx, yy))

    memberships = [[Membership() for x in range(xx[0].size)] for y in range(yy[0].size)]
    data = [xx, yy, zz, memberships]

    centers = np.zeros((NUMBER_OF_GAUSSIAN, 2))

    # fill by function
    x_rand = np.random.uniform(MIN_X, MAX_X, NUMBER_OF_GAUSSIAN)
    y_rand = np.random.uniform(MIN_Y, MAX_Y, NUMBER_OF_GAUSSIAN)

    g_xx, g_xy = np.random.uniform(0, MAX_GXX, NUMBER_OF_GAUSSIAN), np.zeros(NUMBER_OF_GAUSSIAN)
    g_yx, g_yy = np.zeros(NUMBER_OF_GAUSSIAN), np.random.uniform(0, MAX_GYY, NUMBER_OF_GAUSSIAN)

    centers[:, INDEX_X] = x_rand
    centers[:, INDEX_Y] = y_rand

    for i in range(NUMBER_OF_GAUSSIAN):
        print("%f %f | %f %f | %f %f " % (x_rand[i], y_rand[i], g_xx[i], g_xy[i], g_yx[i], g_yy[i]))
        arrange_gaussian(x_rand[i], y_rand[i], g_xx[i], g_xy[i], g_yx[i], g_yy[i], i,
                         np.random.random() > THRESHOLD_SUBTRACTION, data)

    for idx in range(0, xx[0].size):
        for idy in range(0, yy[1].size):
            mem = data[INDEX_MEM][idx][idy]
            total_sum = mem.sum_membership()

            print("%f %f %f " % (data[INDEX_X][idx][idy], data[INDEX_Y][idx][idy], data[INDEX_Z][idx][idy]), end='')

            for mbs, cl in zip(mem.membership, mem.cluster):
                print("%d %f " % (cl, (mbs / total_sum)), end='')

            print()

    # Clustering part
    #
    # =============================================================================================
    X = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))

    centroids, labels = meanshift(X, 0.5)

    # Statistics
    #
    # =============================================================================================
    # Calculate distance from center to the closest Gauss center
    dist_cluster_to_function = cdist(centroids[:, [0, 1]], centers, metric=distance)
    closest_center = np.argmin(dist_cluster_to_function, axis=1)

    dist_sum = 0
    for i in range(0, closest_center.size):
        dist_sum += dist_cluster_to_function[i, closest_center[i]]

    silhouette = metrics.silhouette_score(X, labels, metric=distance)
    calinski_harabaz = metrics.calinski_harabaz_score(X, labels)
    davies_bouldin = metrics.davies_bouldin_score(X, labels)

    # Calculate membership of points to from clusters / Gauss functions

    # Etc etc
    print("Distances: \n ===========================================================")
    print("Cluster to Center of function: %f " % dist_sum)

    print("Silhouette: %f " % silhouette)
    print("Calinski: %f " % calinski_harabaz)
    print("Davies bouldin: %f " % davies_bouldin)

    # Visualize
    #
    # =============================================================================================
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.contourf(xx, yy, zz)

    for cl in centers:
        ax1.scatter(cl[INDEX_X], cl[INDEX_Y], c='r', marker='+')

    xc = centroids[:, 0]
    yc = centroids[:, 1]

    ax1.scatter(xc, yc, c='m', marker='^')

    mngr = plt.get_current_fig_manager()
    geom = mngr.window.geometry()
    x, y, dx, dy = geom.getRect()
    mngr.window.setGeometry(-800, 100, dx, dy)

    # from left, from bottom, from right, from height
    text_box = TextBox(plt.axes([0.1, 0.05, 0.8, 0.075]), 'Bandwidth', initial='')
    text_box.on_submit(calculate_clustering)

    plt.figure()
    ax = plt.axes(projection='3d')

    # points original
    ax.scatter(xx, yy, zz, c=zz.reshape(-1))

    mngr = plt.get_current_fig_manager()
    geom = mngr.window.geometry()
    x, y, dx, dy = geom.getRect()
    mngr.window.setGeometry(-800, 400, dx, dy)

    plt.show()
