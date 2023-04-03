using Base: AbstractVecOrTuple
using Tensors
using DataStructures: DefaultDict

struct EinExpr
    head::Vector{Symbol}
    args::Vector{Union{EinExpr,Tensor}}

    function EinExpr(inputs, output=mapreduce(labels, symdiff, inputs))
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

    # annotate connections of indices
    edges = DefaultDict{Symbol,Set{UInt}}(() -> Set{UInt}())
    for input in expr.args
        for index in labels(input)
            push!(edges[index], objectid(input))
        end
    end

    # compute dual of `edges` dictionary
    dual = DefaultDict{Set{UInt},Vector{Symbol}}(() -> Vector{Symbol}())
    for (index, neighbours) in edges
        length(neighbours) < 2 && continue
        push!(dual[neighbours], index)
    end

    # filter out open indices
    return filter(dual) do (neighbours, inds)
               length(neighbours) >= 2
           end |> values |> collect
end

function Base.string(expr::EinExpr; recursive::Bool=false)
    !recursive && return "$(join(map(x -> string.(labels(x)) |> join, expr.args), ","))->$(string.(labels(expr)) |> join)"
end