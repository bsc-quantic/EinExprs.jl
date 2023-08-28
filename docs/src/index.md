# EinExprs.jl

`EinExprs` is a Julia package that provides `EinExpr`s: symbolic expressions representing a Einstein summation. These summations may be used to represent contraction paths of large tensor networks.

It is a complete redesign of [`OptimizedEinsum`](https://github.com/bsc-quantic/OptimizedEinsum.jl), which indeed was a Julia fork of [`opt_einsum`](https://github.com/dgasmith/opt_einsum). It is design to work with [`Tenet`](https://github.com/bsc-quantic/Tenet.jl) but thanks to the modular structure, it can be used without it.

1. [Einsum Expressions](@ref)
2. [Optimizers](@ref exhaustive_optimizer)
3. [Resource counting](@ref)
4. [Slicing](@ref)
5. [Alternatives](@ref)

## Planned features

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
