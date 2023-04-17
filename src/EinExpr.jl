using Base: AbstractVecOrTuple
using Tensors
using DataStructures: DefaultDict

struct EinExpr
    head::NTuple{N,Symbol} where {N}
    args::Vector{Any}

    function EinExpr(inputs, output=mapreduce(labels, symdiff, inputs))
        # TODO checks: same dim for index, valid indices
        output = Tuple(output)
        inputs = collect(inputs)
        new(output, inputs)
    end
end

Base.:(==)(a::EinExpr, b::EinExpr) = a.head == b.head && a.args == b.args

"""
    labels(expr::EinExpr[, all=false])

Return the indices of the `Tensor` resulting from contracting `expr`. If `all=true`, return a list of all the involved indices.
"""
function Tensors.labels(expr::EinExpr; all::Bool=false)
    !all && return expr.head
    return mapreduce(collect ∘ labels, vcat, expr.args) |> unique |> Tuple
end

"""
    size(expr::EinExpr[, index])

Return the size of the `Tensor` resulting from contracting `expr`. If `index` is specified, return the size of such index.
"""
Base.size(expr::EinExpr) = tuple((size(expr, i) for i in labels(expr))...)

function Base.size(expr::EinExpr, i::Symbol)
    target = findfirst(input -> i ∈ labels(input), expr.args)
    isnothing(target) && throw(KeyError(i))

    return size(expr.args[target], i)
end

"""
    select(expr, i)

Return the child elements (i.e. `Tensor`s or `EinExpr`s) that contain `i` indices.
"""
select(expr::EinExpr, i) = filter(∋(i) ∘ labels, expr.args)
select(expr::EinExpr, i::Base.AbstractVecOrTuple) = ∩(Iterators.map(j -> select(expr, j), i)...)

"""
    neighbours(expr, i)

Return the indices neighbouring to `i`.
"""
neighbours(expr::EinExpr, i) = setdiff(∪(labels.(select(expr, i))...), (i,))
neighbours(expr::EinExpr, i::Base.AbstractVecOrTuple) = setdiff(∪(labels.(select(expr, i))...), i)

"""
    path(expr::EinExpr)

Transform `expr` into a contraction path.
"""
path(expr::EinExpr) = map(suminds, Iterators.filter(x -> x isa EinExpr, expr))

"""
    suminds(expr[, parallel=false])

Indices of summation of an `EinExpr`.
"""
function suminds(expr::EinExpr; parallel::Bool=false)
    !parallel && return setdiff(labels(expr, all=true), labels(expr)) |> collect

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
           end |> values .|> collect |> collect
end

"""
    sum!(expr, indices)

Explicit, in-place sum over `indices`.

See also: [`sum`](@ref), [`suminds`](@ref).
"""
function Base.sum!(expr::EinExpr, inds)
    i = .!isdisjoint.((inds,), labels.(expr.args))

    subargs = splice!(expr.args, findall(i))
    subinds = labels.(subargs)
    subsuminds = setdiff(∩(subinds...), expr.head)
    subhead = setdiff(Iterators.flatten(subinds), subsuminds)

    pushfirst!(expr.args, EinExpr(subargs, subhead))
    return expr
end

"""
    sum(expr, indices)

Explicit sum over `indices`.

See [`sum!`](@ref) for inplace modification.
"""
function Base.sum(expr::EinExpr, inds::Union{Symbol,AbstractVecOrTuple{Symbol}})
    i = .!isdisjoint.((inds,), labels.(expr.args))

    subinds = labels.(expr.args[findall(i)])
    subsuminds = setdiff(∩(subinds...), expr.head)
    suboutput = setdiff(Iterators.flatten(subinds), subsuminds)

    return EinExpr((
            EinExpr(expr.args[findall(i)], suboutput),
            expr.args[findall(.!i)]...,
        ), expr.head)
end

Base.sum(inputs::Union{Tensor,EinExpr}...; inds=mapreduce(labels, symdiff, inputs)) = EinExpr(inputs, inds)

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
