module EinExprsChainRulesCoreExt

using EinExprs
using ChainRulesCore

for f in [
    :head,
    :args,
    :nargs,
    :inds,
    :branches,
    :leaves,
    :suminds,
    :parsuminds,
    :einexpr,
    :sumtraces,
    :indshistogram,
    :hyperinds,
    :neighbours,
    :select,
]
    @eval @non_differentiable EinExprs.$f(::Any...)
end

end