abstract type Optimizer end

function einexpr end

einexpr(T::Type{<:Optimizer}, args...; kwargs...) = einexpr(T(; kwargs...), args...)
einexpr(config::Optimizer, output, inputs, size_dict) = einexpr(config, ...)

include("Exhaustive.jl")
include("Greedy.jl")