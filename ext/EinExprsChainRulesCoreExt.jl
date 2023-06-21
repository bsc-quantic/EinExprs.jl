module EinExprsChainRulesCoreExt

if isdefined(Base, :get_extension)
    using EinExprs
else
    using ..EinExprs
end

using ChainRulesCore
using Tensors: Tensor, contract, labels, nonunique

ChainRulesCore.ProjectTo(expr::EinExpr) = ProjectTo{EinExpr}(; head=expr.head, args=map(ProjectTo, expr.args))
(project::ChainRulesCore.ProjectTo{EinExpr})(dx) = EinExpr(map(((proj_i, dx_i),) -> proj_i(dx_i), zip(project.args, dx.args)), project.head)

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
            expr = EinExpr(partials, labels(e.args[i]))
            tensor = contract(expr)

            # insert singleton dimensions on summed indices
            data = reshape(parent(tensor), map(labels(e.args[i])) do label
                label ∈ labels(tensor) ? size(tensor, label) : 1
            end...)

            # repeat content on summed indices
            data = repeat(data, map(size(data), size(e.args[i])) do size_data, size_orig
                size_data == 1 ? size_orig : 1
            end...)

            tensor = Tensor(data, labels(e.args[i]))

            # set offdiagonal elements to zero if Dirac delta is present
            for index in nonunique(collect(labels(tensor)))
                repeats = count(==(index), labels(tensor))

                for slice in Iterators.filter(!allequal, Iterators.product(repeat([1:size(tensor, index)], repeats)...))
                    offdelta = reduce(slice, init=tensor) do acc, i
                        selectdim(acc, index, i)
                    end
                    broadcast!(() -> zero(eltype(data)), offdelta)
                end
            end

            return tensor
        end

        return f̄, Tangent{EinExpr}(args=c̄)
    end

    return c, contract_pullback
end

end