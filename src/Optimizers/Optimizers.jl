abstract type Optimizer end

function einexpr end

einexpr(T::Type{<:Optimizer}, args...; kwargs...) = einexpr(T(; kwargs...), args...)
einexpr(config::Optimizer, expr, sizedict) = sum(einexpr.((config,), [SizedEinExpr(exp, sizedict) for exp in [comp for comp in components(expr)]]))

include("Naive.jl")
include("Exhaustive.jl")
include("Greedy.jl")
include("KaHyPar.jl")
