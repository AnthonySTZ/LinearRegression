import numpy as np


def gradient_descent_runner(
    points, initial_b, initial_m, learning_rate, num_iterations
):
    pass


def compute_error_for_line_given_points(b, m, points):
    error = 0.0  # Initialize error to 0
    for pt in points:
        x, y = pt
        predicted_y = m * x + b
        error += (y - predicted_y) ** 2  # Squared for positive and punish large errors

    return error / float(len(points))  # Average error across all points


def compute_linear_regression(filename):

    # Collect data
    points = np.genfromtxt(filename, delimiter=",")
    print(type(points))

    # Define Hyperparameters
    learning_rate = 0.0001  # How fast the model converge
    num_iterations = 1000
    # y = mx + b
    initial_b = 0
    initial_m = 0

    # Learning model
    print(
        f"Starting gradient descent at b={initial_b}, m={initial_m}, error={compute_error_for_line_given_points(initial_b, initial_m, points)}"
    )
    # computed_b, computed_m = gradient_descent_runner(
    #     points, initial_b, initial_m, learning_rate, num_iterations
    # )

    # print(
    #     f"Ending point after {num_iterations} iterations, at b={computed_b}, m={computed_m}, error={compute_error_for_line_given_points(computed_b, computed_m, points)}"
    # )
