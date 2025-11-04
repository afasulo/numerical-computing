/*
 * gradient_descent.c
 *
 * This program demonstrates the Gradient Descent algorithm for
 * finding the minimum of a multivariate function.
 *
 * We will minimize the quadratic function:
 * f(x) = 0.5 * x^T * A * x - b^T * x
 *
 * The gradient of this function is:
 * grad(f(x)) = A*x - b
 *
 * The update rule for gradient descent is:
 * x_new = x_old - alpha * grad(f(x_old))
 * x_new = x_old - alpha * (A*x_old - b)
 *
 * The minimum of this function is the solution to Ax = b.
 *
 * This implementation uses OpenMP to parallelize the gradient
 * calculation (the matrix-vector product A*x) and the
 * vector update (x_new = ...).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Define the size of the system
#define N 3

// Set algorithm parameters
#define MAX_ITERATIONS 1000
#define TOLERANCE 1e-6
#define ALPHA 0.1 // Learning rate (This is crucial and problem-dependent)

// Helper function to print a vector
void print_vector(double vec[N]) {
    printf("[ ");
    for (int i = 0; i < N; i++) {
        printf("%.6f ", vec[i]);
    }
    printf("]\n");
}

int main() {
    // Define the problem Ax = b
    // We use a symmetric positive-definite matrix A
    double A[N][N] = {
        {10.0, 1.0, 1.0},
        {1.0, 10.0, 1.0},
        {1.0, 1.0, 10.0}
    };
    double b[N] = {12.0, 12.0, 12.0};
    // Solution x = {1, 1, 1}

    // Solution vector
    double x[N] = {0.0, 0.0, 0.0}; // Initial guess

    // Gradient vector
    double gradient[N];
    // A*x product vector
    double Ax[N];

    int iteration = 0;
    double error = TOLERANCE + 1.0; // Ensure loop runs

    printf("Starting Gradient Descent...\n");
    printf("Initial guess x(0): ");
    print_vector(x);
    printf("--------------------------------------\n");

    while (iteration < MAX_ITERATIONS && error > TOLERANCE) {
        // --- 1. Calculate the gradient: grad = A*x - b ---

        // Parallelize the matrix-vector product: Ax = A * x
        // Each thread computes one element of the 'Ax' vector.
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            Ax[i] = 0.0;
            for (int j = 0; j < N; j++) {
                Ax[i] += A[i][j] * x[j];
            }
        }

        // Compute gradient = Ax - b and calculate L2 norm (error)
        // This loop is sequential for the reduction, but could
        // be parallelized with an 'omp parallel for reduction'
        // for very large N.
        error = 0.0;
        for (int i = 0; i < N; i++) {
            gradient[i] = Ax[i] - b[i];
            error += gradient[i] * gradient[i]; // sum of squares
        }
        error = sqrt(error); // L2 norm of the gradient

        // --- 2. Update the solution: x = x - alpha * gradient ---

        // This update step is also perfectly parallel.
        // Each component x[i] can be updated independently.
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            x[i] = x[i] - ALPHA * gradient[i];
        }

        iteration++;

        // Calculate the "cost" or "loss" f(x)
        // f(x) = 0.5 * x^T * (Ax) - b^T * x
        // This is just for monitoring, not required by the algorithm
        double cost = 0.0;
        double bT_x = 0.0;
        double xT_Ax = 0.0;
        for(int i=0; i<N; i++) {
            bT_x += b[i] * x[i];
            xT_Ax += x[i] * Ax[i];
        }
        cost = 0.5 * xT_Ax - bT_x;


        if (iteration % 5 == 0 || iteration == 1) {
             printf("Iter %4d | Gradient Norm (Error): %.6f | Cost: %.6f\n",
                    iteration, error, cost);
        }
    }

    printf("--------------------------------------\n");
    if (error <= TOLERANCE) {
        printf("Convergence reached in %d iterations.\n", iteration);
        printf("Final Solution (minimum) at:\n");
        print_vector(x);
    } else {
        printf("Failed to converge in %d iterations.\n", MAX_ITERATIONS);
        printf("Current Solution:\n");
        print_vector(x);
    }

    return 0;
}
