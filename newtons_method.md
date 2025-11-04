# Newton's Method for Root Finding

## The Problem

We want to solve f(x) = 0 for some smooth function f: ℝ → ℝ. This is one of those problems that looks deceptively simple but shows up everywhere—optimization (where you're really solving ∇f = 0), solving systems of nonlinear equations, and pretty much any time you need to invert a function numerically.

## Derivation

The core idea is to approximate f locally by its tangent line and solve *that* instead. If we're at some point x_n, the tangent line at (x_n, f(x_n)) is:

```
L(x) = f(x_n) + f'(x_n)(x - x_n)
```

Setting L(x) = 0 and solving for x gives us the next iterate:

```
0 = f(x_n) + f'(x_n)(x_{n+1} - x_n)
x_{n+1} = x_n - f(x_n)/f'(x_n)
```

That's it. That's Newton's method as I understand it.

### Why This Actually Works

The tangent line is just the first-order Taylor expansion of f around x_n. If we kept more terms:

```
f(x) ≈ f(x_n) + f'(x_n)(x - x_n) + (1/2)f''(x_n)(x - x_n)² + ...
```

we'd get higher-order methods (like Halley's method). But first-order is usually good enough and you don't need to compute second derivatives.

## Convergence Analysis

Here's where it gets interesting. Near a simple root (meaning f(r) = 0 and f'(r) ≠ 0), Newton's method has **quadratic convergence**. This means the error roughly squares at each iteration:

```
|e_{n+1}| ≈ C|e_n|²
```

where e_n = x_n - r is the error at step n.

To see why, let's do a "fun" Taylor expansion of f around the true root r:

```
f(x_n) = f(r) + f'(r)(x_n - r) + (1/2)f''(r)(x_n - r)² + O((x_n - r)³)
       = f'(r)e_n + (1/2)f''(r)e_n² + O(e_n³)
```

since f(r) = 0. Similarly:

```
f'(x_n) = f'(r) + f''(r)(x_n - r) + O((x_n - r)²)
        = f'(r) + O(e_n)
```

Now plug these into the Newton iteration:

```
e_{n+1} = x_{n+1} - r
        = x_n - f(x_n)/f'(x_n) - r
        = e_n - [f'(r)e_n + (1/2)f''(r)e_n²] / [f'(r) + O(e_n)]
        ≈ e_n - e_n - (f''(r)/2f'(r))e_n²
        = -(f''(r)/2f'(r))e_n²
```

So the error at step n+1 is proportional to the *square* of the error at step n. This is incredibly fast—if you have 1 correct digit, the next iteration gives you ~2 correct digits, then ~4, then ~8, etc.

## When Things Go Wrong

Newton's method isn't magic though. It can fail in a few ways:

1. **f'(x_n) ≈ 0**: Division by (near) zero. The tangent line is nearly horizontal and the next iterate shoots off the board to infinity.

2. **Bad initial guess**: If you start far from a root, the method might diverge or converge to a different root than you wanted.

3. **Multiple roots**: At a multiple root (where f(r) = f'(r) = 0), convergence is only linear, not quadratic. You need modified Newton for these.

4. **Oscillation**: For some functions, you can get stuck in a cycle. Classic example: f(x) = x³ - 2x + 2 with x₀ = 0 oscillates between 0 and 1.

## Implementation Notes

The C code (`newtons_method.c`) implements this for f(x) = x³ - x - 1. A few things to note:

- We check if |f'(x)| < 10⁻¹² to catch near-horizontal tangents
- Convergence is checked using both step size |x_{n+1} - x_n| and function value |f(x_{n+1})|
- The method is inherently sequential—you can't parallelize the iteration loop because each step depends on the previous one

## Why No Parallelism?

This is a sequential algorithm by nature. The data dependency is:

```
x₁ ← x₀
x₂ ← x₁
x₃ ← x₂
...
```

Each arrow represents computing f and f' at the previous point, which means you can't start computing x_{n+1} until you have x_n. This is different from something like Jacobi iteration where all components of x^(k+1) can be computed in parallel from x^(k).

You *could* try running multiple Newton iterations from different starting points in parallel to find multiple roots, but that's parallelizing across the problem domain, not the algorithm itself.

## Extensions

- **Newton-Raphson in n dimensions**: For solving F(x) = 0 where F: ℝⁿ → ℝⁿ, replace f'/f with J⁻¹F where J is the Jacobian. Now you're solving a linear system at each iteration.

- **Quasi-Newton methods**: Approximate J or J⁻¹ instead of computing it exactly (e.g., Broyden's method). Trades faster iterations for slower convergence.

- **Line search**: Add a damping parameter to prevent overshooting: x_{n+1} = x_n - α·f(x_n)/f'(x_n) where α ∈ (0,1] is chosen adaptively.

## Compilation and Execution

```bash
gcc -o newton newtons_method.c -lm
./newton
```

The `-fopenmp` flag isn't needed since there's nothing to parallelize, but it won't hurt if you include it.
