/*
 * conjugate_gradient.c
 *
 * This program demonstrates the Conjugate Gradient (CG) method for
 * solving a system of linear equations Ax = b, where A is
 * symmetric and positive-definite.
 *
 * The algorithm is parallelized using OpenMP.
 * The key parallel operations are:
 * 1. Matrix-Vector product (A*p)
 * 2. Dot products (r^T*r and p^T*Ap)
 * 3. Vector updates (x = x + a*p, r = r - a*Ap, p = r + b*p)
 *
 * System to solve (same as Jacobi example, A is SPD):
 * 10x +  y +  z = 12
 * x + 10y +  z = 12
 * x +  y + 10z = 12
 *
 * Solution: x=1, y=1, z=1
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Define the size of the system
#define N 3

// Set convergence criteria
#define MAX_ITERATIONS N // CG converges in at most N iterations (in perfect arithmetic)
#define TOLERANCE 1e-6

// Helper function to print a vector
void print_vector(double vec[N]) {
    printf("[ ");
    for (int i = 0; i < N; i++) {
        printf("%.6f ", vec[i]);
    }
    printf("]\n");
}

/*
 * Parallel matrix-vector multiplication: out = A * vec
 */
void mat_vec_mul(double A[N][N], double vec[N], double out[N]) {
    // This loop is parallelized. Each thread handles one or more
    // rows (i) and computes the corresponding element in 'out'.
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        out[i] = 0.0;
        for (int j = 0; j < N; j++) {
            out[i] += A[i][j] * vec[j];
        }
    }
}

/*
 * Parallel dot product: returns vec1^T * vec2
 */
double dot_product(double vec1[N], double vec2[N]) {
    double sum = 0.0;
    // This loop is parallelized using a reduction.
    // Each thread computes a partial sum, and OpenMP
    // combines them safely at the end.
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

/*
 * Parallel vector update: vec1 = vec1 + scalar * vec2
 * (This is a SAXPY-like operation)
 */
void vec_update(double vec1[N], double scalar, double vec2[N]) {
    // This loop is "embarrassingly parallel".
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        vec1[i] += scalar * vec2[i];
    }
}

int main() {
    double A[N][N] = {
        {10.0, 1.0, 1.0},
        {1.0, 10.0, 1.0},
        {1.0, 1.0, 10.0}
    };
    double b[N] = {12.0, 12.0, 12.0};
    double x[N] = {0.0, 0.0, 0.0}; // Initial guess

    double r[N]; // Residual vector (r = b - Ax)
    double p[N]; // Search direction
    double Ap[N]; // Result of A * p

    double rs_old, rs_new, alpha, beta, pAp;
    double error;

    // --- Initialization ---
    // r = b - A*x (Initial residual)
    // Since x=0, r = b.
    // This can be parallelized.
    #pragma omp parallel for
    for(int i=0; i<N; i++) {
        r[i] = b[i];
        p[i] = r[i]; // p_0 = r_0
    }

    // rs_old = r^T * r
    rs_old = dot_product(r, r);
    error = sqrt(rs_old); // Initial error

    printf("Starting Conjugate Gradient...\n");
    printf("Initial guess x(0): ");
    print_vector(x);
    printf("Initial Error (Residual Norm): %.8f\n", error);
    printf("--------------------------------------\n");

    // --- CG Iteration Loop ---
    for (int iter = 0; iter < MAX_ITERATIONS && error > TOLERANCE; iter++) {
        // 1. Calculate A*p
        mat_vec_mul(A, p, Ap);

        // 2. Calculate p^T * A*p
        pAp = dot_product(p, Ap);

        // 3. Calculate alpha = (r^T * r) / (p^T * A*p)
        alpha = rs_old / pAp;

        // 4. Update solution: x = x + alpha * p
        vec_update(x, alpha, p);

        // 5. Update residual: r = r - alpha * A*p
        vec_update(r, -alpha, Ap);

        // 6. Calculate new residual norm: rs_new = r^T * r
        rs_new = dot_product(r, r);
        error = sqrt(rs_new);

        // 7. Calculate beta = rs_new / rs_old
        beta = rs_new / rs_old;

        // 8. Update search direction: p = r + beta * p
        // This must be done carefully.
        #pragma omp parallel for
        for(int i=0; i<N; i++) {
            p[i] = r[i] + beta * p[i];
        }

        // Update rs_old for next iteration
        rs_old = rs_new;

        printf("Iter %2d | Residual Norm: %.8f\n", iter + 1, error);
    }

    printf("--------------------------------------\n");
    if (error <= TOLERANCE) {
        printf("Convergence reached.\n");
        printf("Final Solution:\n");
        print_vector(x);
    } else {
        printf("Failed to converge.\n");
        printf("Current Solution:\n");
        print_vector(x);
    }

    return 0;
}
