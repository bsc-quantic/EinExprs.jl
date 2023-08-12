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

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("../ext/EinExprsMakieExt.jl")
        @require FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000" include(
            "../ext/EinExprsFiniteDifferencesExt.jl",
        )
        # @require ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4" include("../ext/EinExprsChainRulesCoreExt.jl")
    end
end

end
