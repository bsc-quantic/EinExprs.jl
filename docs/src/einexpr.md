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

```@docs
EinExprs.head(::EinExpr)
EinExprs.args
EinExprs.inds
EinExprs.leaves
EinExprs.branches
Base.size(::EinExpr)
EinExprs.suminds
EinExprs.path
```

```@docs
EinExprs.neighbours
EinExprs.select
```

```math
\sum_{i j k l m n o p} A_{mi} B_{ijp} C_{jkn} D_{pkl} E_{mno} F_{ol} = \sum_{i j k l m n p} A_{mi} B_{ijp} C_{jkn} D_{pkl} \sum_o E_{mno} F_{ol}
```

```@docs
Base.sum!(::EinExpr, ::Any)
Base.sum(::EinExpr, ::Union{Symbol, Tuple{Vararg{Symbol}}, AbstractVector{<:Symbol}})
```
