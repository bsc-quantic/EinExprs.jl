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

    # cache flops computation
    flops_best = flops(expr)

    # TODO group indices if they are parallel
    for index in suminds(expr)
        # select tensors containing such index
        targets = filter(∋(index) ∘ labels, expr.args)
        target_inds = labels.(targets)
        contracting_inds = setdiff(∩(target_inds...), expr.head)
        output_inds = setdiff(unique(vcat(target_inds)), contracting_inds)

        # add einsum node of tensor contraction of such index (and its parallels?)
        node = EinExpr(targets, output_inds)
        path = EinExpr([node, filter(∌(index) ∘ labels, expr.args)...], expr.head)

        # prune paths based on flops
        flops(path) >= flops_best && continue

        # TODO prune paths based on memory limit?

        # recurse fixing candidate index
        einexpr(config, candidate)
    end

    # end of path (only reached if flops is best so far)
    return path
end