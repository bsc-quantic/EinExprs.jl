using Base: @kwdef

"""
    Greedy

Greedy contraction path solver. Greedily selects contractions that maximize the metric.
"""
@kwdef struct Greedy <: Optimizer
    choose::Function
    metric::Function = removedsize
end

function einexpr(config::Greedy, expr)
    # TODO memory limit?
end

