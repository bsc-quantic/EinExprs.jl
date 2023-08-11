module EinExprsChainRulesCoreExt

if isdefined(Base, :get_extension)
    using EinExprs
else
    using ..EinExprs
end

using ChainRulesCore
using Tensors: Tensor, contract, labels, nonunique

ChainRulesCore.ProjectTo(expr::EinExpr) = ProjectTo{EinExpr}(; head = expr.head, args = map(ProjectTo, expr.args))
(project::ChainRulesCore.ProjectTo{EinExpr})(dx) =
    EinExpr(project.head, map(((proj_i, dx_i),) -> proj_i(dx_i), zip(project.args, dx.args)))

# TODO recursive call to chain rule of `Tensors.contract` with tensors?
function ChainRulesCore.frule((_, ė), ::typeof(contract), e::EinExpr)
    c = contract(e)

    partials = Iterators.map(((i, arg),) -> EinExpr(head(e), fill(arg, length(args(e)))), enumerate(ė.args))

    ċ = mapreduce(contract, +, partials)

    return c, ċ
end

function ChainRulesCore.rrule(::typeof(contract), e)
    c = contract(e)

    function contract_pullback(ē)
        f̄ = NoTangent()

        c̄ = map(eachindex(args(e))) do i # TODO make it thunkable
            partials = copy(args(e))
            partials[i] = ē

            # compute
            expr = EinExpr(head(args(e)[i]), partials)
            tensor = contract(expr)

            # insert singleton dimensions on summed indices
            data = reshape(
                parent(tensor),
                map(enumerate(head(args(e)[i]))) do (i, index)
                    index ∉ head(tensor) ? 1 : index ∈ head(tensor)[1:i-1] ? 1 : size(tensor, index)
                end...,
            )

            # repeat content on summed indices
            data = repeat(data, map(size(data), size(args(e)[i])) do size_data, size_orig
                size_data == 1 ? size_orig : 1
            end...)

            tensor = Tensor(data, head(args(e)[i]))

            # set offdiagonal elements to zero if Dirac delta is present
            for index in nonunique(collect(head(tensor)))
                repeats = count(==(index), head(tensor))

                for slice in Iterators.filter(!allequal, Iterators.product(repeat([1:size(tensor, index)], repeats)...))
                    offdelta = reduce(slice, init = tensor) do acc, i
                        selectdim(acc, index, i)
                    end
                    broadcast!(() -> zero(eltype(data)), offdelta)
                end
            end

            return tensor
        end

        return f̄, Tangent{EinExpr}(args = c̄)
    end

    return c, contract_pullback
end

end
