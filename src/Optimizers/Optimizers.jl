abstract type Optimizer end

function einexpr end

einexpr(T::Type{<:Optimizer}, args...; kwargs...) = einexpr(T(; kwargs...), args...)
einexpr(config::Optimizer, output, inputs) = einexpr(config, output, inputs)

include("Exhaustive.jl")
include("Greedy.jl")