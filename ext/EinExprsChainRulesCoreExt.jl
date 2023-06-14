module EinExprsChainRulesCoreExt

if isdefined(Base, :get_extension)
    using EinExprs
else
    using ..EinExprs
end

using ChainRulesCore
using Tensors: contract

ChainRulesCore.ProjectTo(expr::EinExpr) = ProjectTo{EinExpr}(; head=expr.head, args=map(ProjectTo, expr.args))
(project::ChainRulesCore.ProjectTo{EinExpr})(dx) = EinExpr(map((proj_i, dx_i) -> proj_i(dx_i), zip(project.args, dx.args)), project.head)

# TODO recursive call to chain rule of `Tensors.contract` with tensors?
function ChainRulesCore.frule((_, ė), ::typeof(contract), e::EinExpr)
    c = contract(e)

    partials = Iterators.map(((i, arg),) -> EinExpr(begin
                args = copy(e.args)
                args[i] = arg
                args
            end, e.head),
        enumerate(ė.args))

    ċ = mapreduce(contract, +, partials)

    return c, ċ
end

function ChainRulesCore.rrule(::typeof(contract), expr)
    c = contract(expr)

    function contract_pullback(Δexpr)
        Δf = NoTangent()
        Δ = map(enumerate(expr.args)) do arg # make it thunkable
            # TODO
        end

        return Δf, Δ...
    end

    return c, contract_pullback
end

end