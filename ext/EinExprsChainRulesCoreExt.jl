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

function ChainRulesCore.rrule(::typeof(contract), e)
    c = contract(e)

    function contract_pullback(ē)
        f̄ = NoTangent()

        c̄ = map(eachindex(e.args)) do i # TODO make it thunkable
            partials = copy(e.args)
            popat!(partials, i)

            # divide primals
            map!(tensor -> 1 ./ tensor, partials, partials)

            # multiply cotangent
            insert!(partials, i, ē)

            # compute
            expr = EinExpr(partials, e.head)
            contract(expr)
        end

        return (f̄, c̄...)
    end

    return c, contract_pullback
end

end