module EinExprs

include("EinExpr.jl")
export EinExpr
export head, args, inds, hyperinds, suminds, parsuminds, collapse!, contractorder, select, neighbours
export Branches, branches, leaves

include("Counters.jl")
export flops, removedsize

include("Slicing.jl")
export findslices, FlopsScorer, SizeScorer

include("Optimizers/Optimizers.jl")
export Optimizer, einexpr
export Exhaustive, Greedy, HyPar

end
