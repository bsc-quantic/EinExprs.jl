abstract type Optimizer end

function einexpr end

einexpr(T::Type{<:Solver}, args...; kwargs...) = einexpr(T(; kwargs...), args...)
einexpr(config::Solver, output, inputs, size_dict) = einexpr(config, ...) # TODO

include("Exhaustive.jl")
include("Greedy.jl")