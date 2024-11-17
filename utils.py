# utils.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Common function to create the Rosenbrock grid
def create_rosenbrock_grid(x_range=(-2, 2), y_range=(-2, 2), resolution=400):
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = 100 * (Y - X**2)**2 + (1 - X)**2
    return X, Y, Z

# Function to plot the Rosenbrock function with paths
def plot_rosenbrock_with_paths(X, Y, Z, paths, colors, start_points, iterations_list, method_name):
    plt.figure(figsize=(10, 8))
    contour_filled = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(contour_filled, label="Function Value")
    plt.plot(1, 1, 'ro', label="Minimum")  # Minimum point
    for i, path in enumerate(paths):
        plt.plot(path[:, 0], path[:, 1], color=colors[i], label=f"Path {i+1}, Iterations: {iterations_list[i]}")
        plt.text(start_points[i][0], start_points[i][1], 
                 f"({start_points[i][0]:.2f}, {start_points[i][1]:.2f})", color=colors[i], fontsize=10, ha='right')
    plt.title(f"{method_name} Convergence on Rosenbrock's Function")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.legend()
    plt.show()

# Function to save the plot in the visualisation folder
def save_plot_to_visualisation(fig, filename):
    fig.savefig(f"visualisations/{filename}.png")  # Ensure correct folder path

