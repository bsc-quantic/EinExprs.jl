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

function einexpr(config::Exhaustive, path, sizedict; cost = BigInt(0))
    metric = Base.Fix2(config.metric, sizedict)

    leader = (; path = einexpr(Naive(), path), cost = mapreduce(metric, +, PreOrderDFS(einexpr(Naive(), path))))
    cache = Dict{ImmutableVector{Symbol,Vector{Symbol}},BigInt}()

    function __einexpr_iterate(path, cost)
        if length(path.args) <= 2
            leader = (; path = path, cost = mapreduce(metric, +, PreOrderDFS(path)))
            return
        end

        for (i, j) in combinations(args(path), 2)
            !config.outer && isdisjoint(head(i), head(j)) && continue
            candidate = sum([i, j], skip = path.head ∪ hyperinds(path))

            # prune paths based on metric
            new_cost = cost + get!(cache, head(candidate)) do
                metric(candidate)
            end
            new_cost >= leader.cost && continue

            new_path = EinExpr(head(path), [candidate, filter(∉([i, j]), args(path))...])
            __einexpr_iterate(new_path, new_cost)
        end
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
