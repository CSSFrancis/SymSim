import numpy as np

from numpy.random import random, choice

from SymSim.sim.cluster import Cluster
from SymSim.utils.rotation_utils import _get_random_2d_rot, _get_random_3d_rot
from SymSim.utils.vector_utils import build_ico


def get_random_cluster(num_clusters,
                       dimensions,
                       radius_range=(.5, 1.0),
                       k_range=(3.5, 4.5),
                       random_rotation=True,
                       symmetry=[2, 4, 6, 10],
                       ):

    rand_r = random(num_clusters) * (radius_range[1] - radius_range[0]) + radius_range[0]
    rand_k = random(num_clusters) * (k_range[1] - k_range[0]) + k_range[0]
    rand_sym = choice(symmetry, num_clusters)
    rand_pos = np.multiply(random((num_clusters, 3)), dimensions)
    if random_rotation:
        rot_3d = _get_random_3d_rot(num_clusters)
        rot_2d = _get_random_2d_rot(num_clusters)
    else:
        rot_2d = [np.eye(3), ] * num_clusters
        rot_3d = [np.eye(3), ] * num_clusters
    cluster_list = []
    for s, r, k, p, two, three in zip(rand_sym, rand_r, rand_k, rand_pos, rot_2d, rot_3d):
        cluster_list.append(Cluster(s, r, k, p, two, three))
    return cluster_list


def get_random_icosahedron(num_clusters,
                           dimensions,
                           radius_range=(.5, 1.0),
                           k_range=(3.5, 4.5),
                           random_rotation=True):
        """This adds in the appropriate symmetries for an icosohedron like structure.
        Parameters
        ------------
        num_clusters: int
            The number of cluster to add
        radius_range: tuple
            The range of radii to randomly choose from
        k_range: tuple
            THe range of k to randomly choose from
        random_rotation: bool
            Random rotate the cluster or not
        symmetry: list
            The list of symmetries to choose from.  Acceptable symmetries are 2,4,6 and 10
        """
        rand_r = random(num_clusters) * (radius_range[1] - radius_range[0]) + radius_range[0]
        rand_k = random(num_clusters) * (k_range[1] - k_range[0]) + k_range[0]
        rand_pos = np.multiply(random((num_clusters,3)), dimensions)
        if random_rotation:
            rot_3d = _get_random_3d_rot(num_clusters)
            rot_2d = _get_random_2d_rot(num_clusters)
        else:
            rot_2d = [np.eye(3), ]*num_clusters
            rot_3d = [np.eye(3), ]*num_clusters
        five_vertexes, three_face, two_edge = build_ico()
        cluster_list =[]
        for r, k, p, two, three in zip(rand_r, rand_k,rand_pos, rot_2d, rot_3d):
            for v in five_vertexes:
                cluster_list.append(Cluster(10,r,k,p,two,three,plane_direction=v))
            for v in three_face:
                cluster_list.append(Cluster(6,r,k,p,two,three,plane_direction=v))
            for v in two_edge:
                cluster_list.append(Cluster(2, r, k, p, two, three, plane_direction=v))
        return cluster_list