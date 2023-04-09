using Base: AbstractVecOrTuple
using Tensors
using DataStructures: DefaultDict

struct EinExpr
    head::NTuple{N,Symbol} where {N}
    args::NTuple{M,Union{EinExpr,Tensor}} where {M}

    function EinExpr(inputs, output=mapreduce(labels, symdiff, inputs))
        # TODO checks: same dim for index, valid indices
        output = Tuple(output)
        inputs = Tuple(inputs)
        new(output, inputs)
    end
end

function Tensors.labels(expr::EinExpr; all::Bool=false)
    !all && return expr.head
    return mapreduce(collect ∘ labels, vcat, expr.args) |> unique |> Tuple
end

path(expr::EinExpr) = vcat([path(i) for i in expr.args if i isa EinExpr]..., suminds(expr, parallel=false))

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
    !parallel && return setdiff(labels(expr, all=true), labels(expr)) |> Tuple

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
           end |> values .|> Tuple |> Tuple
end

"""
    sum(expr, indices)

Explicit sum over `indices`.
"""
function Base.sum(expr::EinExpr, inds)
    i = .!isdisjoint.((inds,), labels.(expr.args))

    subinds = labels.(expr.args[findall(i)])
    subsuminds = setdiff(∩(subinds...), expr.head)
    suboutput = setdiff(Iterators.flatten(subinds), subsuminds)

    return EinExpr((
            EinExpr(expr.args[findall(i)], suboutput),
            expr.args[findall(.!i)]...,
        ), expr.head)
end

function Base.string(expr::EinExpr; recursive::Bool=false)
    !recursive && return "$(join(map(x -> string.(labels(x)) |> join, expr.args), ","))->$(string.(labels(expr)) |> join)"
end

# Iteration interface
Base.IteratorSize(::Type{EinExpr}) = Base.HasLength()
Base.length(expr::EinExpr) = sum(arg -> arg isa EinExpr ? length(arg) : 1, expr.args) + 1
Base.IteratorEltype(::Type{EinExpr}) = Base.HasEltype()
Base.eltype(::EinExpr) = Union{<:Tensor,EinExpr}

# TODO only return `EinExpr`s?
function Base.iterate(expr::EinExpr, state=1)
    isnothing(state) && return nothing

    # iterate child level
    i, j... = state
    it = iterate(expr.args, i)

    # return itself on last iteration
    isnothing(it) && return expr, nothing

    # recurse iteration
    (next, statenext) = it

    # if `next` is a Tensor, return directly
    !(next isa EinExpr) && return next, statenext

    next, j = if isempty(j)
        iterate(next)
    else
        iterate(next, j)
    end

    # if j === nothing, expr.args iteration has finished
    isnothing(j) && return next, i + 1

    return next, (i, j...)
end
