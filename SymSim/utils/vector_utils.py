import numpy as np


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def build_ico():
    five_vertexes = np.array([[0,1,1.9], [0,-1,1.9],[0,1,-1.9], [0,-1,-1.9],
                              [1,1.9,0], [-1,1.9,0],[1,-1.9,0], [-1,-1.9,0],
                              [1.9,0,1], [-1.9,0,1],[1.9,0,-1], [-1.9,0,-1]])
    n = np.linalg.norm(five_vertexes, axis=1)
    five_vertexes = np.divide(five_vertexes, n[:,np.newaxis])
    two_edge = []
    for v1 in five_vertexes:
        for v2 in five_vertexes:
            #two_edge.append(np.linalg.norm(np.array(v1)-np.array(v2)))
            #print(np.linalg.norm(np.array(v1)-np.array(v2)))
            if np.abs(np.linalg.norm(np.array(v1)-np.array(v2)) - 1.08) <.3 :
               two_edge.append(v1+v2)
    n = np.linalg.norm(two_edge, axis=1)
    two_edge = np.divide(two_edge, n[:,np.newaxis])
    two_edge = np.unique(two_edge, axis=0)
    three_face = []
    for v1 in five_vertexes:
        for v2 in five_vertexes:
            for v3 in five_vertexes:
                n1 = np.abs(np.linalg.norm(np.array(v1) - np.array(v2))-1.08)
                n2 = np.abs(np.linalg.norm(np.array(v1) - np.array(v3))-1.08)
                n3 = np.abs(np.linalg.norm(np.array(v2) - np.array(v3))-1.08)
                if  n1 < .3 and n2 < .3 and  n3 < .3:
                    three_face.append(v1+v2+v3)
    n = np.linalg.norm(three_face, axis=1)
    three_face = np.divide(three_face, n[:,np.newaxis])
    three_face = np.unique(three_face,axis=0)
    return five_vertexes, three_face, two_edge


def build_ico_positions():
    positions = [[0.0, 0.0, 0.0],
                 [0.0, 0.85065080835204, 0.5257311121191336],
                 [0.0, 0.85065080835204, -0.5257311121191336],
                 [0.0, -0.85065080835204, 0.5257311121191336],
                 [0.0, -0.85065080835204, -0.5257311121191336],
                 [0.5257311121191336, 0.0, 0.85065080835204],
                 [-0.5257311121191336, 0.0, 0.85065080835204],
                 [0.5257311121191336, 0.0, -0.85065080835204],
                 [-0.5257311121191336, 0.0, -0.85065080835204],
                 [0.85065080835204, 0.5257311121191336, 0.0],
                 [0.85065080835204, -0.525731112119133, 0.0],
                 [-0.85065080835204, 0.5257311121191336, 0.0],
                 [-0.85065080835204, -0.5257311121191336, 0.0]]
    return np.array(positions)


def get_ico_edges(positions, shell=2):
    edges = []
    atoms = []
    for i, v1 in enumerate(positions):
        for j, v2 in enumerate(positions):
            dist = np.abs(np.linalg.norm(np.array(v1)-np.array(v2)))
            if shell*1.06 > dist > shell:
                atoms.append(sorted([i+1, j+1]))
                edges.append((v1+v2)/2)
    edges = np.unique(edges, axis=0)
    atoms = np.unique(atoms, axis=0)
    return atoms, edges


def get_ico_faces(positions, shell_dist=1):
    vectors = []
    faces = []
    for i, v1 in enumerate(positions):
        for j, v2 in enumerate(positions):
            for k, v3 in enumerate(positions):
                n1 = np.abs(np.linalg.norm(np.array(v1) - np.array(v2))-shell_dist*1.06)/shell_dist
                n2 = np.abs(np.linalg.norm(np.array(v1) - np.array(v3))-shell_dist*1.06)/shell_dist
                n3 = np.abs(np.linalg.norm(np.array(v2) - np.array(v3))-shell_dist*1.06)/shell_dist
                if n1 < .3 and n2 < .3 and n3 < .3:
                    vectors.append((v1+v2+v3)/3)
                    faces.append(sorted([i+1, j+1, k+1]))
    print(faces)
    faces = np.unique(faces, axis=0)
    vectors = np.unique(vectors, axis=0)
    return faces, vectors


def _get_angle_between(v1, v2):
    unit_vector1 = v1 / np.linalg.norm(vector1)
    unit_vector2 = v2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    return np.arccos(dot_product)  # angle in radian

def _get_distance_from_sphere(x,y,radius):
    print(np.shape(x))
    return radius - np.sqrt(radius**2-x**2-y**2)

def _get_distance_from_sphere_delta(x,y,radius, deltaxy, deltaz):
    magnitudes = (x**2+y**2)**0.5
    x_displacement = (x*deltaxy)
    y_displacement = (y*deltaxy)
    new_x = x+x_displacement
    new_y = y+y_displacement
    return (radius - np.sqrt(radius**2-new_x**2-new_y**2))

def _get_deflection_from_convergence(convergence_angle, radius):
    """Get some max deflection in x,y, and z for some convergence angle
    and Ewald Sphere radius

    Parameters
    ------------
    convergence_angle: float
        The Convergence angle in mRad
    radius:
        The radius of the Ewald Sphere (1/wavelength)

    Returns
    --------
    magnitude_x,y
    """
    magnitude_deflection = np.sin(convergence_angle/2000)*radius
    delta_z = np.sin(convergence_angle/2000)*magnitude_deflection
    magnitude_xy = np.cos(convergence_angle/2000)*magnitude_deflection
    return magnitude_xy,magnitude_xy, delta_z
