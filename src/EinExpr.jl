using Base: AbstractVecOrTuple
using Tensors
using DataStructures: DefaultDict

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

function Base.size(expr::EinExpr, i::Symbol)
    target = findfirst(input -> i ∈ labels(input), expr.args)
    isnothing(target) && throw(KeyError(i))

    return size(expr.args[target], i)
end

Base.size(expr::EinExpr) = tuple((size(expr, i) for i in labels(expr))...)

"""
    suminds(expr)

Indices of summation of an `EinExpr`.
"""
function suminds(expr::EinExpr; parallel::Bool=false)
    !parallel && return setdiff(labels(expr, all=true), labels(expr))

    # compute connections of indices
    edges = DefaultDict{Symbol,Vector{UInt}}(() -> UInt[])
    for input in expr.args
        for index in labels(input)
            push!(edges[index], objectid(input))
        end
    end

    # compute dual of dictionary
    dual = DefaultDict{Vector{UInt},Set{Symbol}}(() -> Set{Symbol}())
    for (index, neighbours) in edges
        length(neighbours) < 2 && continue
        push!(dual[neighbours], index)
    end

    return filter(>=(2) ∘ length, collect(values(dual)))
end
