module EinExprs

include("EinExpr.jl")
export EinExpr
export head, args, inds, leaves, branches, collapse!, suminds, contractorder, select

include("Counters.jl")
export flops, removedsize

include("Slicing.jl")
export findslices, FlopsScorer, SizeScorer

include("Optimizers/Optimizers.jl")
export Optimizer, einexpr
export Exhaustive, Greedy

end
