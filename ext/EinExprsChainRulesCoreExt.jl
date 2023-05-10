module EinExprsChainRulesCoreExt

if isdefined(Base, :get_extension)
    using EinExprs
else
    using ..EinExprs
end

using ChainRulesCore

function ChainRulesCore.frule((_, Δexpr), ::typeof(contract), expr)
    c = contract(expr)
    Δc = contract(Δexpr)

    return c, Δc
end

end