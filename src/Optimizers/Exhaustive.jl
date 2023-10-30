using Base: @kwdef
using Combinatorics

@doc raw"""
    Exhaustive(; outer = false)

Exhaustive contraction path optimizers. It guarantees to find the optimal contraction path but at a large cost.

# Keywords

- `outer` instructs to consider outer products (aka tensor products) on the search for the optimal contraction path. It rarely provides an advantage over only considering inner products and thus, it is `false` by default.

!!! warning
    The functionality of `outer = true` has not been yet implemented.

# Implementation

The algorithm has a ``\mathcal{O}(n!)`` time complexity if `outer = true` and ``\mathcal{O}(\exp(n))`` if `outer = false`.
"""
@kwdef struct Exhaustive <: Optimizer
    metric::Function = flops
    outer::Bool = false
end

function einexpr(config::Exhaustive, path; cost = BigInt(0))
    leader = Ref{NamedTuple{(:path, :cost),Tuple{EinExpr,BigInt}}}((;
        path = einexpr(Naive(), path),
        cost = mapreduce(config.metric, +, Branches(einexpr(Naive(), path), inverse = true), init = BigInt(0))::BigInt,
    ))
    cache = Dict{Vector{Symbol},BigInt}()
    __einexpr_exhaustive_it(path, cost, config.metric, config.outer, leader, cache)
    return leader[].path
end

function __einexpr_exhaustive_it(path, cost, metric, outer, leader, cache)
    if length(path.args) == 1
        # remove identity einsum (i.e. "i...->i...")
        path = path.args[1]

        leader[] = (; path, cost = mapreduce(metric, +, Branches(path, inverse = true), init = BigInt(0))::BigInt)
        return
    end

    for (i, j) in combinations(args(path), 2)
        !outer && isdisjoint(head(i), head(j)) && continue
        candidate = sum([i, j], skip = path.head ∪ hyperinds(path))

        # prune paths based on metric
        new_cost = cost + get!(cache, head(candidate)) do
            metric(candidate)
        end
        new_cost >= leader[].cost && continue

        new_path = EinExpr(head(path), [candidate, filter(∉([i, j]), args(path))...])
        __einexpr_exhaustive_it(new_path, new_cost, metric, outer, leader, cache)
    end
end
