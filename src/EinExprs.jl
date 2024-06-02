module EinExprs

include("Utils.jl")

include("EinExpr.jl")
export EinExpr
export head, args, inds, hyperinds, suminds, parsuminds, collapse!, contractorder, select, neighbours
export Branches, branches, Leaves, leaves

include("SizedEinExpr.jl")
export SizedEinExpr

include("Counters.jl")
export flops, removedsize

include("Slicing.jl")
export findslices, FlopsScorer, SizeScorer

include("Canonization.jl")

include("Optimizers/Optimizers.jl")
export Optimizer, einexpr
export Exhaustive, Greedy, HyPar

using PackageExtensionCompat
function __init__()
    @require_extensions
end

end
