using Base: @kwdef

@kwdef struct HyPar <: Optimizer
    parts::Int = 2
    imbalance::Float32 = 0.03
    stop::Function = <=(2) ∘ length ∘ Base.Fix2(getproperty, :args)
    configuration::Union{Nothing,Symbol,String} = nothing
    edge_scaler::Function = Base.Fix1(*, 1000) ∘ Int ∘ round ∘ log2
    vertex_scaler::Function = Base.Fix1(*, 1000) ∘ Int ∘ round ∘ log2
    seed::Int = 0
end
