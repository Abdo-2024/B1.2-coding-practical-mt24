# Optimization of the Rosenbrock Function

This project focuses on applying and visualizing various optimization techniques to minimize the Rosenbrock function. The project uses Python to implement optimization algorithms and visualize their performance.

## Table of Contents
1. [Project Description](#project-description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Optimization Algorithms](#optimization-algorithms)
    - [Gradient Descent](#gradient-descent)
    - [Newton's Method](#newtons-method)
    - [Gauss-Newton](#gauss-newton)
    - [Nelder-Mead Simplex](#nelder-mead-simplex)
5. [Visualization](#visualization)
6. [Requirements](#requirements)
7. [License](#license)

## Project Description

This project demonstrates the optimization of the Rosenbrock function using several popular optimization algorithms. These algorithms include:
- **Gradient Descent**
- **Newton's Method**
- **Gauss-Newton**
- **Nelder-Mead Simplex**

The goal is to visualize the behavior of each optimization method in finding the minimum of the Rosenbrock function and analyze their convergence and performance.

The `task1.py` script visualizes the Rosenbrock function in both 2D and 3D, while `task2.py` implements the first three optimization algorithms. The `task3.py` script applies the Nelder-Mead Simplex method.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Abdo-2024/B1.2-coding-practical-mt24.git

2. Navigate to the project directory:

cd rosenbrock-optimization

3. Set up a Python virtual environment (optional but recommended):

python3 -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`

4. Install the required dependencies:

pip install -r requirements.txt

## Usage

After setting up the environment, you can run the scripts in the following order:

1. Visualize the Rosenbrock Function: To generate 2D and 3D visualizations of the Rosenbrock function, run task1.py:

python task1.py

2. Optimize the Rosenbrock Function: To apply the optimization algorithms (Gradient Descent, Newton's Method, and      Gauss-Newton), run task2.py:

python task2.py

3. Apply the Nelder-Mead Simplex Optimization: To use the Nelder-Mead Simplex optimization algorithm, run task3.py:

python task3.py

4. Run All Tasks from main.py: To run all tasks (visualization and optimization algorithms) sequentially, execute the main.py script:

python main.py

## Optimization Algorithms

### Gradient Descent

Gradient Descent is a first-order optimization algorithm that updates parameters iteratively in the direction of the negative gradient of the function. It is used here to optimize the Rosenbrock function by updating the parameters using the gradient.

### Newton's Method

Newton's Method is a second-order optimization technique that uses both the gradient and the Hessian matrix (second derivatives) to find the minimum. In this project, it is used to optimize the Rosenbrock function.

### Gauss-Newton

The Gauss-Newton method is a variation of Newton's method that is specifically designed for non-linear least squares problems. In this project, it is applied to minimize the Rosenbrock function by approximating the Hessian matrix.

### Nelder-Mead Simplex

The Nelder-Mead method is a direct search method that does not require the gradient or Hessian of the function. It uses the geometry of the function's space to find the optimal point.

## Visualization

1.  The 2D and 3D visualizations of the Rosenbrock function are created using matplotlib. These visualizations help to see the function's landscape and observe how optimization methods approach the minimum.
2.  The optimization path for each algorithm is plotted to visualize how the algorithms converge toward the minimum of the Rosenbrock function.

## Requirements

To run this project, you will need the following Python libraries:

    numpy
    matplotlib
    scipy

You can install them using the following command:

pip install numpy matplotlib scipy


## License

This project is licensed under the MIT License - see the LICENSE file for details.