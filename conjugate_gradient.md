# Conjugate Gradient Method

## Why Gradient Descent Isn't Enough

Gradient descent on a quadratic function f(x) = (1/2)x^T Ax - b^T x has a problem: it zigzags. Each step minimizes f along the gradient direction, but then the next step often cancels part of that progress because the new gradient points in a nearly perpendicular direction.

For an ill-conditioned matrix (i.e has a large condition number κ), you need O(κ) iterations to converge. That's... not great.


Conjugate gradient (CG) corrects this by choosing *smarter* search directions. Instead of repeatedly descending along gradients, we descend along **A-conjugate** (or A-orthogonal) directions. The payoff: convergence in at most n steps for an n×n system, and O(√κ) iterations in practice.

## The Central Idea: A-Conjugacy

Two vectors p and q are A-conjugate (or A-orthogonal) if:

```
p^T A q = 0
```

This is like orthogonality, but with respect to the inner product ⟨p,q⟩_A = p^T A q instead of the standard inner product p^T q.

Why does this matter? Suppose we have n mutually A-conjugate search directions p₀, p₁, ..., p_{n-1}. Then we can write the exact solution as:

```
x* = Σᵢ αᵢ pᵢ
```

for some coefficients αᵢ. And here's the magic trick: we can compute each αᵢ independently by minimizing f along that direction.

## The Algorithm (High Level)

Starting from x⁰ = 0 (or any initial guess) and r⁰ = b - Ax⁰:

```
1. Set p⁰ = r⁰ (first search direction is the gradient)
2. For k = 0, 1, 2, ...
   a. Step size: αₖ = (rᵏ^T rᵏ) / (pᵏ^T A pᵏ)
   b. Update solution: x^(k+1) = xᵏ + αₖ pᵏ
   c. Update residual: r^(k+1) = rᵏ - αₖ A pᵏ
   d. If converged, stop
   e. Improvement direction: βₖ = (r^(k+1)^T r^(k+1)) / (rᵏ^T rᵏ)
   f. New search direction: p^(k+1) = r^(k+1) + βₖ pᵏ
```

The directions p⁰, p¹, p², ... are automatically A-conjugate. That's not obvious at all, but it falls out of the math.

## Derivation: Where Do These Formulas Come From?

Let's derive the CG algorithm from scratch. We want to minimize f(x) along direction pᵏ starting from xᵏ:

```
min_α f(xᵏ + α pᵏ)
```

Taking the derivative with respect to α and setting it to zero:

```
d/dα f(xᵏ + α pᵏ) = ∇f(xᵏ + α pᵏ)^T pᵏ = 0
```

At the optimum α = αₖ:

```
∇f(x^(k+1))^T pᵏ = 0
(A x^(k+1) - b)^T pᵏ = 0
```

Since the residual r^(k+1) = b - A x^(k+1), this says:

```
(r^(k+1))^T pᵏ = 0
```

So the new residual is orthogonal to the search direction. Cool.

Now, how do we choose the next search direction p^(k+1)? We want it to be A-conjugate to all previous directions. The clever insight: take p^(k+1) to be the residual r^(k+1) plus a correction that makes it A-orthogonal to pᵏ:

```
p^(k+1) = r^(k+1) - βₖ pᵏ
```

To find βₖ, impose A-conjugacy:

```
(p^(k+1))^T A pᵏ = 0
(r^(k+1) - βₖ pᵏ)^T A pᵏ = 0
βₖ = (r^(k+1)^T A pᵏ) / (pᵏ^T A pᵏ)
```

But we can simplify this. Since r^(k+1) = rᵏ - αₖ A pᵏ:

```
r^(k+1)^T A pᵏ = (rᵏ - αₖ A pᵏ)^T A pᵏ
               = rᵏ^T A pᵏ - αₖ (A pᵏ)^T A pᵏ
```

Now, from the definition of αₖ and the fact that rᵏ^T pᵏ = rᵏ^T rᵏ (I'm skipping some algebra here, but you can verify this using pᵏ = rᵏ + βₖ₋₁ p^(k-1) and orthogonality of residuals), we get:

```
rᵏ^T A pᵏ = (rᵏ^T rᵏ) / αₖ
```

Substituting back:

```
r^(k+1)^T A pᵏ = (rᵏ^T rᵏ) / αₖ - (rᵏ^T rᵏ) = -βₖ (pᵏ^T A pᵏ) (rᵏ^T rᵏ) / αₖ
```

Wait, this is getting messy. Let me use a different approach.

Actually, there's a cleaner formula that you can derive using the conjugacy relations:

```
βₖ = (r^(k+1)^T r^(k+1)) / (rᵏ^T rᵏ)
```

This is the Fletcher-Reeves formula. It's equivalent to the one above but only requires residual norms, not matrix-vector products.

## Why CG Works: Krylov Subspaces

Here's the deeper theory. Define the Krylov subspace:

```
𝒦ₖ = span{r⁰, Ar⁰, A²r⁰, ..., A^(k-1)r⁰}
```

CG has the property that:
1. The iterate xᵏ lies in the affine subspace x⁰ + 𝒦ₖ
2. xᵏ minimizes f over that subspace
3. The residual rᵏ is orthogonal to 𝒦ₖ

This is huge. It means CG is implicitly building an orthonormal basis for 𝒦ₖ (the search directions) and solving a small k×k optimization problem at each step.

Since 𝒦ₙ = ℝⁿ for an n×n matrix, CG finds the exact solution in at most n steps (in exact arithmetic). In practice, rounding errors screw this up, but you still converge quickly.

## Convergence Rate

In practice, CG converges long before n iterations. The error decreases as:

```
||x* - xᵏ||_A ≤ 2 ((√κ - 1)/(√κ + 1))^k ||x* - x⁰||_A
```

where ||z||_A = √(z^T A z) is the A-norm and κ = λ_max/λ_min is the condition number.

Compare to gradient descent: O(κ) iterations → CG: O(√κ) iterations. For κ = 10,000 that's 10,000 vs. 100 iterations. That's why people actually use CG.

## Preconditioning

If A is badly conditioned, even O(√κ) can be too slow. The solution: preconditioning.

Instead of solving Ax = b, solve:

```
M⁻¹ A x = M⁻¹ b
```

where M ≈ A but M⁻¹ is cheap to compute. If M is a good approximation, M⁻¹A has a much smaller condition number.

Common preconditioners:
- **Jacobi**: M = diag(A)
- **Incomplete Cholesky**: M ≈ LL^T with sparse factors
- **Multigrid**: M is a V-cycle (very effective for PDEs)

Preconditioned CG (PCG) is what you'd actually use in production code. But vanilla CG is already a huge improvement over gradient descent.

## Parallelization

The C implementation parallelizes three operations:

1. **Matrix-vector product** (A*p): Each row can be computed independently.
   ```c
   #pragma omp parallel for
   for (int i = 0; i < N; i++) {
       Ap[i] = Σⱼ A[i][j] * p[j];
   }
   ```

2. **Dot products** (r^T r, p^T Ap): Use parallel reduction.
   ```c
   #pragma omp parallel for reduction(+:sum)
   for (int i = 0; i < N; i++) {
       sum += r[i] * r[i];
   }
   ```

3. **Vector updates** (x = x + α*p, etc.): Embarrassingly parallel.
   ```c
   #pragma omp parallel for
   for (int i = 0; i < N; i++) {
       x[i] += alpha * p[i];
   }
   ```

For n = 3, parallelization overhead probably dominates. But for large sparse systems (n = 10⁶), this scales well.

The *iteration* loop itself is sequential—each iteration depends on the previous one. But that's fine because most of the work is in the parallel operations within each iteration.

## Implementation Notes

The code uses several helper functions:
- `mat_vec_mul`: Computes y = Ax with `#pragma omp parallel for`
- `dot_product`: Computes x^T y with reduction
- `vec_update`: SAXPY operation (y = y + α*x)

These are the BLAS-level primitives that CG needs. In a real scientific code, you'd use an actual BLAS library (like OpenBLAS or MKL) which has highly optimized versions of these.

## When to Use CG

CG is great when:
- A is symmetric positive definite (SPD)
- A is large and sparse (so you can do matrix-vector products quickly)
- You don't need the exact solution, just an approximation

CG is overkill for:
- Small dense systems (just use Cholesky factorization)
- Indefinite systems (need MINRES or GMRES instead)
- Very ill-conditioned systems without a good preconditioner

For our test case (3×3 diagonally dominant matrix), CG is total overkill. But it's a nice proof of concept.

## Compilation and Execution

```bash
gcc -o cg conjugate_gradient.c -fopenmp -lm
./cg
```

The output shows the residual norm at each iteration. For the test system, CG converges in 3 iterations (as expected, since n = 3).
