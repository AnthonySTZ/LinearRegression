import numpy as np


def gradient_descent_runner(
    points, starting_b, starting_m, learning_rate, num_iterations
):
    # Starting m and b
    b = starting_b
    m = starting_m

    # Gradient descent
    for _ in range(num_iterations):
        # gradient step
        b, m = gradient_step(b, m, points, learning_rate)

    return b, m


def gradient_step(b_current, m_current, points, learning_rate):

    b_gradient = 0
    m_gradient = 0
    N = float(len(points))  # Number of points

    for pt in points:
        x, y = pt

        # Direction respect to b and m
        # Computing partial derivatives of our error function

        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))

    b_new = b_current - (b_gradient * learning_rate)
    m_new = m_current - (m_gradient * learning_rate)
    return b_new, m_new


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
    computed_b, computed_m = gradient_descent_runner(
        points, initial_b, initial_m, learning_rate, num_iterations
    )

    print(
        f"Ending point after {num_iterations} iterations, at b={computed_b}, m={computed_m}, error={compute_error_for_line_given_points(computed_b, computed_m, points)}"
    )
