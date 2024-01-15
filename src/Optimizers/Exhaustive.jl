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
    # metric = Base.Fix2(config.metric, path.size)
    leader = Ref((;
        path = einexpr(Naive(), path),
        cost = mapreduce(config.metric, +, Branches(einexpr(Naive(), path), inverse = true), init = BigInt(0))::BigInt,
    ))
    __einexpr_exhaustive_it(path, cost, Val(config.metric), config.outer, leader)
    return leader[].path
end

function __einexpr_exhaustive_it(
    path::SizedEinExpr{L},
    cost,
    @specialize(metric::Val{Metric}),
    outer,
    leader;
    cache = Dict{Vector{L},BigInt}(),
    hashyperinds = !isempty(hyperinds(path)),
) where {L,Metric}
    if nargs(path) <= 2
        #= mapreduce(metric, +, Branches(path, inverse = true), init = BigInt(0))) =#
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
        __einexpr_exhaustive_it(new_path, new_cost, metric, outer, leader; cache, hashyperinds)
    end
end
