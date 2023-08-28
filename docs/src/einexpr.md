# Einsum Expressions

Let's take an arbitrary tensor network like the following.

```@raw html
<img class="invert-on-dark" src="../assets/tensor-network.excalidraw.svg" alt="An arbitrary tensor network"/>
```

This graph is a graphical notation equivalent to the following equation.

```math
\sum_{i j k l m n o p} A_{mi} B_{ijp} C_{jkn} D_{pkl} E_{mno} F_{ol}
```

A naïve implementation of this equation is easy to implement.

```julia
result = zero(reduce(promote_type, eltype.([A,B,C,D,E,F])))
for (i,j,k,l,m,n,o,p) in Iterators.product(1:I, 1:J, 1:K, 1:L, 1:M, 1:N, 1:O, 1:P)
    result += A[m,i] * B[i,j,p] * C[j,k,n] * D[p,k,l] * E[m,n,o] * F[o,l]
end
```

But it has a cost of ``\prod_\alpha \dim(\alpha)`` where ``\alpha \in \{i,j,k,l,m,n,o,p\}`` which is of ``\mathcal{O}(\exp(n))`` time complexity.

## Basic utilities

The `EinExpr` type has a stable definition: it's secure to access its fields using the dot-syntax (e.g. `path.head`), but we recommend to use the following methods instead. They provide a much more stable API and can be used in a functional manner.

```@docs
EinExprs.head
EinExprs.args
Base.size(::EinExpr)
```

Some other useful methods are:

```@docs
Base.ndims(::EinExpr)
EinExprs.inds
EinExprs.suminds
EinExprs.parsuminds
EinExprs.contractorder
EinExprs.select
EinExprs.neighbours
```

## Construction by summation

One option for constructing `EinExpr`s manually is to use the [`sum`](@ref) methods. For example, imagine that we have the following tensor equation and we want to contract first tensors ``E_{mno}`` and ``F_{ol}``. The resulting equation would be equivalent to adding a summatory to ``E`` and ``F`` as written in the right-hand side.

```math
\sum_{i j k l m n o p} A_{mi} B_{ijp} C_{jkn} D_{pkl} E_{mno} F_{ol} = \sum_{i j k l m n p} A_{mi} B_{ijp} C_{jkn} D_{pkl} \sum_o E_{mno} F_{ol}
```

In `EinExprs`, we advocate for code that it's _almost_ as easy as writing math. As such, one can write `sum([E, F])` to create a new `EinExpr` where common indices are contracted or `sum!(path, :o)` for the in-place version where ``E`` and ``F`` are children of `path`.

```@docs
Base.sum!(::EinExpr, ::Any)
Base.sum(::EinExpr, ::Union{Symbol, Tuple{Vararg{Symbol}}, AbstractVector{<:Symbol}})
```

In order to reverse the operation and unfix the contraction, the user may call the [`collapse!`](@ref) function.

```@docs
EinExprs.collapse!
```

## `AbstractTrees` integration

`EinExpr` type integrates with the [`AbstractTrees`](https://github.com/JuliaCollections/AbstractTrees.jl) package in order to implement some of the tree-traversing algorithms.
The interface is public and thus any user can use it to implement their own methods.

For example, the `AbstractTrees.Leaves` function returns an iterator through the leaves of any tree; i.e. the initial tensors in our case. We implement the `Branches` function in order to walk through the non-terminal nodes; i.e. the intermediate tensors.

```@docs
EinExprs.Branches
```

`EinExprs` exports a variant of these methods which return collections.

```@docs
EinExprs.leaves
EinExprs.branches
```
