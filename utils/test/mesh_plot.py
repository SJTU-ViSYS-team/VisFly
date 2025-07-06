import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_triangle_mesh(vertices, faces):
    """
    Plots a triangle mesh given vertices and faces.

    :param vertices: A 2D numpy array of shape (n, 3) where n is the number of vertices.
    :param faces: A 2D numpy array of shape (m, 3) where m is the number of faces, each face is defined by 3 vertex indices.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    vertices = np.array(vertices)
    faces = np.array(faces).reshape(-1,3)
    # Create a Poly3DCollection for the mesh
    mesh = Poly3DCollection(vertices[faces], alpha=0.5, edgecolor='k')

    ax.add_collection3d(mesh)

    # Set limits
    ax.set_xlim([np.min(vertices[:, 0]), np.max(vertices[:, 0])])
    ax.set_ylim([np.min(vertices[:, 1]), np.max(vertices[:, 1])])
    ax.set_zlim([np.min(vertices[:, 2]), np.max(vertices[:, 2])])

    plt.show()


def plot_rectangle_mesh(vertices, faces):
    """
    Plots a triangle mesh given vertices and faces.

    :param vertices: A 2D numpy array of shape (n, 3) where n is the number of vertices.
    :param faces: A 2D numpy array of shape (m, 3) where m is the number of faces, each face is defined by 3 vertex indices.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    vertices = np.array(vertices)
    faces = np.array(faces).reshape(-1,4)
    # Create a Poly3DCollection for the mesh
    mesh = Poly3DCollection(vertices[faces], alpha=0.5, edgecolor='k')

    ax.add_collection3d(mesh)

    # Set limits
    ax.set_xlim([np.min(vertices[:, 0]), np.max(vertices[:, 0])])
    ax.set_ylim([np.min(vertices[:, 1]), np.max(vertices[:, 1])])
    ax.set_zlim([np.min(vertices[:, 2]), np.max(vertices[:, 2])])

    plt.show()