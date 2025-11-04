# Gradient Descent for Quadratic Optimization

## The Setup

We're minimizing a quadratic function of the form:

```
f(x) = (1/2)x^T A x - b^T x
```

where A ∈ ℝⁿˣⁿ is symmetric positive definite (SPD) and b ∈ ℝⁿ. This might look like a toy problem, but it shows up constantly:

- Least squares regression: minimize ||Ax - b||²
- Solving linear systems: min f(x) is equivalent to solving Ax = b
- Newton's method for general optimization: each step solves a quadratic subproblem

Plus, understanding the quadratic case gives you all the intuition for gradient descent on general smooth functions.

## The Algorithm

The gradient of f is:

```
∇f(x) = Ax - b
```

Gradient descent moves in the direction of steepest descent:

```
x^(k+1) = x^(k) - α∇f(x^(k)) = x^(k) - α(Ax^(k) - b)
```

where α > 0 is the step size (or "learning rate" if you're from the ML world).

### Why This Direction?

At any point x, the gradient ∇f(x) points in the direction of steepest *ascent*. So -∇f(x) points in the direction of steepest *descent*. A first-order Taylor expansion confirms this:

```
f(x + d) ≈ f(x) + ∇f(x)^T d
```

This is minimized (to first order) when d = -α∇f(x) for some small α > 0.

## Convergence Analysis

Here's where quadratic functions are nice: we can analyze everything exactly.

Let x* = A⁻¹b be the minimizer (so ∇f(x*) = 0). Define the error e^(k) = x^(k) - x*. Then:

```
e^(k+1) = x^(k+1) - x*
        = x^(k) - α(Ax^(k) - b) - x*
        = x^(k) - x* - α(Ax^(k) - Ax*)
        = e^(k) - αAe^(k)
        = (I - αA)e^(k)
```

So the error evolves as:

```
e^(k) = (I - αA)^k e^(0)
```

### When Does This Converge?

We need ||(I - αA)^k|| → 0 as k → ∞. Since A is SPD, it has real positive eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₙ > 0, and the spectral norm satisfies:

```
||(I - αA)|| = max_i |1 - αλᵢ|
```

For convergence, we need |1 - αλᵢ| < 1 for all i, which gives:

```
0 < α < 2/λ₁
```

So the step size must be less than twice the reciprocal of the largest eigenvalue.

### Convergence Rate

The error decreases by a factor of ||(I - αA)|| at each step:

```
||e^(k)|| ≤ ||(I - αA)||^k ||e^(0)||
```

The optimal step size is α* = 2/(λ₁ + λₙ), which gives:

```
||(I - α*A)|| = (λ₁ - λₙ)/(λ₁ + λₙ) = (κ - 1)/(κ + 1)
```

where κ = λ₁/λₙ is the **condition number** of A.

This is the key insight: convergence depends on how "round" the level curves of f are. If κ ≈ 1 (A ≈ cI), the function is spherical and gradient descent converges in one step. If κ >> 1 (A is ill-conditioned), convergence is slow.

For κ large, we get roughly:

```
||e^(k)|| ≤ (1 - 2/κ)^k ||e^(0)||
```

So you need O(κ) iterations to reduce the error by a constant factor. This is **linear convergence**, much slower than Newton's quadratic convergence.

## The Learning Rate Problem

The step size α is crucial. Too small and you converge slowly. Too large and you diverge.

In the implementation (`gradient_descent.c`), α = 0.1. For the test matrix A = diag(10,10,10) + ones(3,3), the eigenvalues are roughly:
- λ₁ ≈ 12 (largest)
- λ₃ ≈ 9 (smallest)

So the convergence bound is α < 2/12 ≈ 0.167. Our choice α = 0.1 is safe but not optimal (optimal would be around 0.095).

### Adaptive Step Sizes

In practice, you don't know the eigenvalues ahead of time. Common approaches:

1. **Line search**: At each iteration, choose αₖ to minimize f(x^(k) - αₖ∇f(x^(k)))

2. **Backtracking**: Start with a large α and shrink it until you get sufficient decrease (Armijo condition)

3. **Adam/RMSprop**: Adapt α per-coordinate using gradient statistics (very popular in deep learning)

For quadratics, the optimal step size per iteration is:

```
αₖ = (r^(k)^T r^(k)) / (r^(k)^T A r^(k))
```

where r^(k) = ∇f(x^(k)) is the residual. This is the "steepest descent" method and converges faster than fixed α.

## Parallelization

Gradient descent parallelizes well within each iteration:

1. **Compute Ax**: Matrix-vector multiplication. Each component (Ax)ᵢ = Σⱼ Aᵢⱼxⱼ is independent.

2. **Compute gradient**: Just subtract b from Ax (trivially parallel).

3. **Update x**: Each component x_i^(k+1) = x_i^(k) - α·∇f(x^(k))_i is independent.

The C implementation uses OpenMP with `#pragma omp parallel for` on all three loops. On a machine with p cores, you'd expect roughly a p× speedup for large n (for n = 3, overhead probably dominates).

## Why Not Just Use Gradient Descent?

For quadratic problems, there are much better methods:

- **Conjugate Gradient**: O(√κ) iterations instead of O(κ), and no need to tune α
- **Direct methods**: For small-to-medium systems, just factor A = LL^T and solve in O(n³) time
- **Preconditioned methods**: Transform the problem to reduce κ

For general nonlinear optimization, gradient descent is a baseline but you'd usually prefer:
- **BFGS/L-BFGS**: Quasi-Newton methods with superlinear convergence
- **Nesterov acceleration**: Momentum-based methods that achieve O(√κ) rate
- **Newton-CG**: Newton with CG for the subproblem

That said, gradient descent is simple, robust, and works even when you can't compute Hessians. It's the workhorse of machine learning for a reason.

## Connection to Solving Linear Systems

Notice that minimizing f(x) = (1/2)x^T Ax - b^T x is equivalent to solving Ax = b:

```
∇f(x) = 0  ⟺  Ax - b = 0  ⟺  Ax = b
```

So gradient descent on f is really an iterative method for solving linear systems. The update becomes:

```
x^(k+1) = x^(k) - α(Ax^(k) - b)
```

This is just a stationary iteration with M = I/α:

```
x^(k+1) = x^(k) + M⁻¹(b - Ax^(k))
```

Compare this to Jacobi (M = D) or Gauss-Seidel (M = D + L). Gradient descent uses the trivial preconditioner M = I/α.

## Compilation and Execution

```bash
gcc -o gradient gradient_descent.c -fopenmp -lm
./gradient
```

The output shows the gradient norm (which is ||∇f(x^(k))|| = ||Ax^(k) - b||, i.e., the residual) and the function value at each iteration. Convergence is declared when the gradient norm falls below the tolerance.
