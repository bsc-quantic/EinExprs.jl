abstract type Optimizer end

function einexpr end

einexpr(T::Type{<:Optimizer}, args...; kwargs...) = einexpr(T(; kwargs...), args...)
einexpr(config::Optimizer, expr) = einexpr(config, expr)

include("Naive.jl")
include("Exhaustive.jl")
include("Greedy.jl")
include("KaHyPar.jl")
