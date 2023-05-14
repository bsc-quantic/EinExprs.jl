# EinExprs.jl

`EinExprs` is a Julia package that provides `EinExpr`s: symbolic expressions representing a Einstein summation.

It is a complete rewrite of [`OptimizedEinsum`](https://github.com/bsc-quantic/OptimizedEinsum.jl), which indeed was a Julia fork of [`opt_einsum`](https://github.com/dgasmith/opt_einsum). It is design to work with [`Tenet`](https://github.com/bsc-quantic/Tenet.jl) but this is not a strong requirement and can be adapted to your needs.

## Roadmap

- Automatic Differentiation
- Tensor cutting aka slicing
- Batch contraction
- GPU offloading
- Distributed execution
- Optimizers
  - Hypergraph Partitioning
  - Branch & Bound
  - Dynamic Programming
- Subtree reconfiguration
- Resource counting
  - _Array structure_-aware analysis
- Execution planning
- Permutation order optimization
- Compilation
