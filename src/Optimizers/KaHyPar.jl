using Base: @kwdef

@kwdef struct HyPar <: Optimizer
    imbalances::StepRange{Int, Int} = 130:130
end
