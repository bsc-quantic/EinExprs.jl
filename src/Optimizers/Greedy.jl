using Base: @kwdef

"""
    Greedy

Greedy contraction path solver. Greedily selects contractions that maximize the heuristic.
"""
@kwdef struct Greedy <: Optimizer
    choose::Function
    heuristic::Function
end

function einexpr(config::Greedy, path)
    # TODO memory limit?
end

