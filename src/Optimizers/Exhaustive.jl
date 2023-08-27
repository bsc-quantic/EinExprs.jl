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
    leader = (; path = einexpr(Naive(), path), cost = mapreduce(config.metric, +, PreOrderDFS(einexpr(Naive(), path))))
    cache = Dict{Vector{ImmutableVector{Symbol,Vector{Symbol}}},BigInt}()

    function __einexpr_iterate(path, cost)
        if length(path.args) <= 2
            leader = (; path = path, cost = mapreduce(config.metric, +, PreOrderDFS(path)))
            return
        end

        for (i, j) in combinations(args(path), 2)
            !config.outer && isdisjoint(head(i), head(j)) && continue
            candidate = sum([i, j])

            # prune paths based on metric
            new_cost = cost + get!(() -> config.metric(candidate), cache, head.(candidate.args))
            new_cost >= leader.cost && continue

            new_path = EinExpr(head(path), [candidate, filter(∉([i, j]), args(path))...])
            __einexpr_iterate(new_path, new_cost)
        end
    end

    __einexpr_iterate(path, cost)

    return leader.path
end