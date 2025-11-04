/*
 * newtons_method.c
 *
 * This program demonstrates Newton's method for finding the
 * root of a single-variable function f(x).
 *
 * Example function:
 * f(x) = x^3 - x - 1
 *
 * Derivative:
 * f'(x) = 3x^2 - 1
 *
 * The root is approximately 1.3247.
 *
 * Note: This algorithm is sequential by nature. The (n+1)th
 * iteration depends directly on the result of the nth iteration,
 * so parallelization of the main loop is not possible.
 */

#include <stdio.h>
#include <math.h>

// Define the function f(x)
double f(double x) {
    return x * x * x - x - 1.0;
}

// Define the derivative f'(x)
double f_prime(double x) {
    return 3.0 * x * x - 1.0;
}

/**
 * @brief Performs Newton's method to find a root.
 *
 * @param x0 Initial guess.
 * @param tolerance Convergence tolerance.
 * @param max_iterations Maximum number of iterations.
 */
void newtons_method(double x0, double tolerance, int max_iterations) {
    double x_old = x0;
    double x_new;
    double f_x;
    double f_prime_x;
    double step;
    int iteration = 0;

    printf("Starting Newton's method with x0 = %.6f\n", x0);
    printf("---------------------------------------------\n");
    printf("Iter |    x_n    |   f(x_n)   |   f'(x_n)  |   Step Size\n");
    printf("---------------------------------------------\n");

    while (iteration < max_iterations) {
        f_x = f(x_old);
        f_prime_x = f_prime(x_old);

        // Check for division by zero (tangent is horizontal)
        if (fabs(f_prime_x) < 1e-12) {
            printf("Error: Derivative is zero at x = %.6f\n", x_old);
            return;
        }

        // Newton's iteration formula
        step = f_x / f_prime_x;
        x_new = x_old - step;

        printf("%4d | %.6f | %.6e | %.6e | %.6e\n",
               iteration, x_old, f_x, f_prime_x, fabs(step));

        // Check for convergence
        // We check if the step size is small OR if f(x) is close to zero
        if (fabs(step) < tolerance || fabs(f(x_new)) < tolerance) {
            printf("---------------------------------------------\n");
            printf("Convergence reached in %d iterations.\n", iteration + 1);
            printf("Root is approximately: %.8f\n", x_new);
            return;
        }

        // Prepare for next iteration
        x_old = x_new;
        iteration++;
    }

    printf("---------------------------------------------\n");
    printf("Failed to converge in %d iterations.\n", max_iterations);
    printf("Current value: %.8f\n", x_new);
}

int main() {
    // Set initial parameters
    double initial_guess = 1.5; // A good starting guess
    double tolerance = 1e-7;
    int max_iterations = 100;

    newtons_method(initial_guess, tolerance, max_iterations);

    return 0;
}
