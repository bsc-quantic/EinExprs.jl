module EinExprsChainRulesCoreExt

if isdefined(Base, :get_extension)
    using EinExprs
else
    using ..EinExprs
end

using ChainRulesCore

end