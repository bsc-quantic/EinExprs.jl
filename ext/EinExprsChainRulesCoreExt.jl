module EinExprsChainRulesCoreExt

using EinExprs
using ChainRulesCore

@non_differentiable einexpr(::Any...)

end