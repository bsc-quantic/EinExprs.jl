using Base: @kwdef

"""
    Exhaustive([outer = false])

Exhaustive contraction path optimizers. It guarantees to find the optimal contraction path but at the cost of ``\\mathcal{O}(n!)`` time complexity.
"""
@kwdef struct Exhaustive <: Optimizer
    outer::Bool = false
end

function einexpr(config::Exhaustive, expr)
    config.outer && throw("Exhaustive search with outer-products is not implemented yet")

    # TODO group indices if they are parallel
    # TODO cache flops computation
    # TODO type annotation in `suminds(expr, parallel=true)::Vector{Vector{Symbol}}` for type-inference?
    # NOTE `for index in suminds(expr)` is better for debugging
    return reduce(suminds(expr, parallel=true), init=expr) do leader, inds
        # select tensors containing such inds
        targets = filter(x -> !isdisjoint(labels(x), inds), leader.args)

        subinds::Vector{Base.AbstractVecOrTuple{Symbol}} = labels.(targets)
        subsuminds = setdiff(∩(subinds...), leader.head)
        suboutput = setdiff(Iterators.flatten(subinds), subsuminds)

        # add einsum node of tensor contraction of such inds (and its parallels?)
        candidate = EinExpr([
                EinExpr(targets, suboutput),
                filter(∌(inds) ∘ labels, leader.args)...
            ], leader.head)

        # prune paths based on flops
        flops(candidate) >= flops(leader) && return leader

        # TODO prune paths based on memory limit?

        # recurse fixing candidate index
        return einexpr(config, candidate)
    end
end
