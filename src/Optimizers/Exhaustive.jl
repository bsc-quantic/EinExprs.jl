using Base: @kwdef

"""
    Exhaustive([outer = false])

Exhaustive contraction path optimizers. It guarantees to find the optimal contraction path but at the cost of ``\\mathcal{O}(n!)`` time complexity.
"""
@kwdef struct Exhaustive <: Optimizer
    outer::Bool = false
end

function einexpr(config::Exhaustive, einexpr)
    config.outer && throw("Exhaustive search with outer-products is not implemented yet")

    flops_best = Ref(typemax(Int128))
    __einexpr_exhaustive(einexpr, flops_best)
end

function __einexpr_exhaustive(path, flops_best)
    # TODO depth-first search of paths
    for indices in uncontractedinds(path)
        # TODO get (intermediate) tensors containing such index

        # TODO add einsum node of tensor contraction of such index (and its parallels?)
        candidate = copy(path)
        node = Expr(:call, :einsum, indices)
        push!(candidate, node)

        # prune paths based on flops
        flops(candidate) >= flops_best[] && continue

        # TODO prune paths based on memory limit?

        # recurse fixing candidate index
        __einexpr_exhaustive(candidate, flops_best)
    end

    # end of path (only reached if flops is best so far)
    flops_best[] = flops(path)
    return path
end