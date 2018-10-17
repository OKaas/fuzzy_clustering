import numpy as np  # best of the best
import matplotlib.pyplot as plt  # matlab plot
from scipy.stats import multivariate_normal

INDEX_X = 0
INDEX_Y = 1
INDEX_Z = 2
INDEX_MEM = 3

MIN_X = -1
MAX_X = 1

MIN_Y = -1
MAX_Y = 1

STEP_X = 0.1
STEP_Y = 0.1


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

def arrange_gaussian(p_x, p_y, p_cl, p_data):
    rv = multivariate_normal([p_x, p_y], [[0.1, 0], [0, 0.1]])

    add = rv.pdf(pos)

    # add gaussian
    p_data[INDEX_Z] += add

    for idx in range(0, add[0].size):
        for idy in range(0, add[1].size):
            p_data[INDEX_MEM][idx][idy].add_influence(p_cl, add[idx][idy])


class DataGenerator:
    def __init__(self):
        # arrange data ( or np.mgrid(MIN_X, ) )
        x = np.arange(MIN_X, MAX_X, STEP_X)
        y = np.arange(MIN_X, MAX_X, STEP_Y)

        # create matrix for data
        xx, yy = np.meshgrid(x, y)

        # Z value is not know yet -> will be changed by functions
        zz = xx.copy()
        zz.fill(0)
        pos = np.dstack((xx, yy))

        memberships = [[Membership() for x in range(xx[0].size)] for y in range(yy[0].size)]
        data = [xx, yy, zz, memberships]

        # fill by function
        arrange_gaussian(1, 1, 1, data)
        arrange_gaussian(-1, -1, 2, data)
        arrange_gaussian(0, 0, 3, data)
        arrange_gaussian(-0.3, 0.3, 4, data)
        arrange_gaussian(0.6, -0.75, 5, data)

        for idx in range(0, xx[0].size):
            for idy in range(0, yy[1].size):
                mem = data[INDEX_MEM][idx][idy]
                total_sum = mem.sum_membership()

                print("%f %f %f " % (data[INDEX_X][idx][idy], data[INDEX_Y][idx][idy], data[INDEX_Z][idx][idy]), end='')

                for mbs, cl in zip(mem.membership, mem.cluster):
                    print("%d %f " % (cl, (mbs / total_sum)), end='')

                print()

    plt.figure()
    plt.contourf(xx, yy, zz)
    plt.scatter(1, 1, c='r', marker='+')
    plt.scatter(-1, -1, c='r', marker='+')
    plt.scatter(0, 0, c='r', marker='+')
    plt.scatter(-0.3, 0.3, c='r', marker='+')
    plt.scatter(0.6, -0.75, c='r', marker='+')
    plt.show()
