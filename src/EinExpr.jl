using Base: AbstractVecOrTuple
using Tensors

struct EinExpr
    head::Vector{Symbol}
    args::Vector{Union{EinExpr,Tensor}}

    function EinExpr(inputs::AbstractVecOrTuple{Union{EinExpr,Tensor}}, output=mapreduce(labels, symdiff, inputs))
        # TODO checks: same dim for index, valid indices
        output = collect(output)
        new(output, inputs)
    end
end

function Tensors.labels(expr::EinExpr; all::Bool=false)
    !all && return expr.head
    return mapreduce(collect ∘ labels, vcat, expr.args) |> unique
end

Base.ndims(expr::EinExpr) = length(labels(expr))

function Base.size(expr::EinExpr, i::Symbol)
    target = findfirst(input -> i ∈ labels(input), expr.args)
    isnothing(target) && throw(KeyError(i))

    return size(expr.args[target], i)
end

Base.size(expr::EinExpr) = tuple((size(expr, i) for i in labels(expr))...)

# uncontractedinds(expr::EinExpr) = ... # TODO

# function flops(expr::EinExpr)
#     flops_children = map(tensor -> begin
#             if tensor isa EinExpr
#                 ... # TODO
#             else # Tensor
#                 0
#             end
#         end, expr.args)

#     length(flops_children) == 1 && return only(flops_children)
#     return ... # TODO
# end