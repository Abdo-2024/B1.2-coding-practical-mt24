import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Defining Rosenbrock Function
def rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2


# Task 1: Visualize Rosenbrock Function
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# 2D Contour Plot
plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=50, cmap="viridis")
plt.plot(1, 1, 'ro')  # Mark the minimum point
plt.colorbar(contour)
plt.title("2D Contour Plot of Rosenbrock's Function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 3D Surface Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor='none', alpha=0.7)
ax.set_title("3D Surface Plot of Rosenbrock's Function")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
ax.scatter(1, 1, rosenbrock(1, 1), color='r', s=50)  # Mark the minimum point
fig.colorbar(surface)
plt.show()
