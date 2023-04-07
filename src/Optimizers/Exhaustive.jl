using Base: @kwdef

"""
    Exhaustive([outer = false])

Exhaustive contraction path optimizers. It guarantees to find the optimal contraction path but at the cost of ``\\mathcal{O}(n!)`` time complexity.
"""
@kwdef struct Exhaustive <: Optimizer
    outer::Bool = false
end

function einexpr(config::Exhaustive, expr; leader=expr)
    config.outer && throw("Exhaustive search with outer-products is not implemented yet")

    if length(suminds(expr, parallel=true)) == 1
        return flops(expr) < flops(leader) ? expr : leader
    end

    # TODO type annotation in `suminds(expr, parallel=true)::Vector{Vector{Symbol}}` for type-inference?
    # NOTE `for index in suminds(expr)` is better for debugging
    for inds in suminds(expr, parallel=true)
        # select tensors containing such inds
        targets = filter(x -> !isdisjoint(labels(x), inds), expr.args)

        subinds = labels.(targets)
        subsuminds = setdiff(∩(subinds...), expr.head)
        suboutput = setdiff(Iterators.flatten(subinds), subsuminds)

        candidate = EinExpr(targets, suboutput)

        # prune paths based on flops
        flops(candidate) >= flops(leader) && continue

        # TODO prune paths based on memory limit?

        # recurse fixing candidate index
        candidate = EinExpr([
                candidate,
                filter(x -> isdisjoint(labels(x), inds), expr.args)...,
            ], expr.head)
        leader = einexpr(config, candidate, leader=leader)
    end

    return leader
end
