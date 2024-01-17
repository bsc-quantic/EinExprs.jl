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

function einexpr(config::Exhaustive, path::SizedEinExpr{L}; cost = BigInt(0)) where {L}
    init_path = einexpr(Naive(), path)
    leader = Ref((;
        path = init_path,
        cost = mapreduce(config.metric, +, Branches(init_path, inverse = true), init = BigInt(0))::BigInt,
    ))
    exhaustive_depthfirst(Val(config.metric), path, cost, config.outer, leader)
    return leader[].path
end

function exhaustive_depthfirst(
    @specialize(metric::Val{Metric}),
    path::SizedEinExpr{L},
    cost,
    outer,
    leader;
    cache = Dict{Vector{L},BigInt}(),
    hashyperinds = !isempty(hyperinds(path)),
) where {L,Metric}
    if nargs(path) <= 2
        leader[] = (; path = path, cost = cost)
        return
    end

    for (i, j) in combinations(path.args, 2)
        !outer && isdisjoint(head(i), head(j)) && continue
        candidate = sum(i, j; skip = hashyperinds ? path.head ∪ hyperinds(path) : path.head)

        # prune paths based on metric
        new_cost = cost + get!(cache, head(candidate)) do
            Metric(SizedEinExpr(candidate, path.size))
        end
        new_cost >= leader[].cost && continue

        new_path = SizedEinExpr(EinExpr(head(path), [candidate, filter(∉([i, j]), path.args)...]), path.size) # sum([candidate, filter(∉([i, j]), args(path))...], skip = path.head)
        exhaustive_depthfirst(metric, new_path, new_cost, outer, leader; cache, hashyperinds)
    end
end
