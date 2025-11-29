# Numerical Computing Portfolio

A rigorous exploration of fundamental algorithms in scientific computing, demonstrating mastery of stability analysis, convergence theory, and computational complexity.

## Core Topics

### Finite Difference Methods & Error Analysis
- **Taylor Series Approximations**: Derived O(h²) centered difference formulas with catastrophic cancellation analysis at the ϵ^(1/3) threshold
- **Optimal Step Size Selection**: Minimized total error E(h) = C₁h² + C₂ϵ/h through calculus of variations, achieving machine precision bounds

### Direct & Iterative Linear Solvers
- **Gaussian Elimination**: Stability analysis of pivot selection strategies; demonstrated catastrophic failure modes for ill-conditioned systems with condition numbers κ(A) > 10¹¹
- **Iterative Methods**: Implemented Jacobi and Gauss-Seidel with spectral radius convergence analysis; contrasted O(n) sparse matrix operations against O(n²) banded LU factorization
- **Conjugate Gradient**: Achieved quadratic convergence for SPD systems, reaching machine precision in ≤n iterations for n×n matrices

### Polynomial Interpolation Theory
- **Lagrange vs. Vandermonde Bases**: Comparative numerical stability analysis; Lagrange formulation circumvents ill-conditioning for degree n > 26
- **Runge's Phenomenon**: Demonstrated divergence with equispaced nodes; mitigated via Chebyshev points xᵢ = cos((2i+1)π/(2n+2)) achieving exponential convergence for analytic functions
- **Piecewise Methods**: Natural cubic splines eliminate high-degree oscillations through C² continuity constraints on subintervals

### Rootfinding Algorithms
- **Newton's Method**: Proved quadratic convergence ‖eₖ₊₁‖ ≤ C‖eₖ‖² for simple roots where f'(x*) ≠ 0; degraded to linear for multiple roots f'(x*) = 0
- **Bisection Method**: Guaranteed linear convergence with eₖ₊₁/eₖ = 1/2; derived break-even iteration counts versus direct methods

### Numerical Integration
- **Composite Quadrature Rules**: Implemented trapezoid and Simpson schemes with h² and h⁴ error bounds respectively
- **Gaussian Quadrature**: Leveraged Legendre polynomial orthogonality to achieve degree 2k-1 exactness with k-point formulas; demonstrated sixth-order convergence O(h⁶) for 3-point rule
- **Adaptive Methods**: Error estimation via Richardson extrapolation and interval subdivision strategies

### Eigenvalue Computation
- **Power Iteration**: Normalized iterative scheme converging to dominant eigenpair at rate |λ₂/λ₁|ᵏ
- **Matrix Deflation**: Rank-1 perturbation B = A - λ₁v⁽¹⁾(v⁽¹⁾)ᵀ/‖v⁽¹⁾‖² for subsequent eigenvalue extraction
- **Numerical Conditioning**: Resolved eigenvalues separated by O(√ϵₘ) where characteristic polynomial methods fail due to floating-point rounding

## Mathematical Foundations

**Convergence Orders**: Rigorously validated through log-log regression: order p satisfies eₖ ≈ Ch^p where C absorbs derivative magnitudes

**Stability Criteria**: Exploited condition number bounds κ(A) = ‖A‖‖A⁻¹‖ to predict solution accuracy δx/‖x‖ ≤ κ(A)δb/‖b‖

**Computational Complexity**: Analyzed asymptotic costs—O(n³) direct factorization versus O(kn) for k iterations of sparse methods with ≤5 nonzeros per row

## Implementation Details

All algorithms implemented in MATLAB with emphasis on:
- Vectorized operations leveraging BLAS/LAPACK optimizations
- Machine epsilon ϵₘ ≈ 2.22×10⁻¹⁶ awareness in stopping criteria
- Comprehensive convergence studies across varying problem dimensions and condition numbers

---

*This repository represents advanced coursework (CS/MATH 375) synthesizing theory from numerical linear algebra, approximation theory, and scientific computing.*
