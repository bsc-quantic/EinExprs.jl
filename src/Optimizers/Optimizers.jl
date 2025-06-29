abstract type Optimizer end

function einexpr end

einexpr(T::Type{<:Optimizer}, args...; kwargs...) = einexpr(T(; kwargs...), args...)
einexpr(config::Optimizer, expr, sizedict) = einexpr(config, SizedEinExpr(expr, sizedict))

function einexpr(path::SizedEinExpr; optimizer)
    path = deepcopy(path)

    # remove inds with dim=1 (shadow inds)
    canonize!(SumGhostInds(), path)

    einexpr(optimizer, path)
end

include("Naive.jl")
include("Exhaustive.jl")
include("Greedy.jl")
include("HyPar.jl")
include("LineGraph.jl")
