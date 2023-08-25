using Base: AbstractVecOrTuple
using DataStructures: DefaultDict
using ImmutableArrays

struct EinExpr
    head::ImmutableVector{Symbol,Vector{Symbol}}
    args::Vector{Union{Tensor,EinExpr}}

    function EinExpr(head, args)
        # TODO checks: same dim for index, valid indices
        head = collect(head)
        args = collect(args)
        new(head, args)
    end
end

"""
    head(path::EinExpr)

Return the indices of the resulting tensor from contracting `path`.

See also: [`inds`](@ref), [`args`](@ref).
"""
head(path::EinExpr) = path.head
head(tensor::Tensor) = tensor.inds

"""
    args(path::EinExpr)

Return the children of the `path`, which correspond to input tensors for the contraction step in the top of the `path`.

See also: [`head`](@ref).
"""
args(path::EinExpr) = path.args
args(::Tensor) = []

"""
    inds(path)

Return all the involved indices in `path`. If a tensor is passed, then it is equivalent to calling [`head`](@ref).

See also: [`head`](@ref).
"""
inds(path::EinExpr) = mapreduce(head, vcat, args(path)) |> unique
inds(tensor::Tensor) = head(tensor)

"""
    leaves(path::EinExpr[, i])

Return the terminal leaves of the `path`, which correspond to the initial input tensors.
If `i` is specified, then only return the ``i``-th tensor.

See also: [`branches`](@ref).
"""
leaves(path) = Iterators.filter(Base.Fix2(isa, Tensor), path) |> first
leaves(path, i) = Iterators.drop(Iterators.filter(Base.Fix2(isa, Tensor), path), i) |> first

"""
    branches(path::EinExpr[, i])

Return the non-terminal branches of the `path`, which correspond to intermediate tensors result of contraction steps.
If `i` is specified, then only return the ``i``-th `EinExpr`.

See also: [`leaves`](@ref).
"""
branches(path) = Iterators.filter(Base.Fix2(isa, EinExpr), path)
branches(path, i) = Iterators.drop(Iterators.filter(Base.Fix2(isa, EinExpr), path), i) |> first

Base.:(==)(a::EinExpr, b::EinExpr) = a.head == b.head && a.args == b.args

"""
    ndims(path::EinExpr)

Return the number of indices of the resulting tensor from contracting `path`.
"""
Base.ndims(path::EinExpr) = length(head(path))

"""
    size(path::EinExpr[, index])

Return the size of the resulting tensor from contracting `path`. If `index` is specified, return the size of such index.
"""
Base.size(path::EinExpr) = tuple((size(path, i) for i in head(path))...)
Base.size(path::EinExpr, i::Symbol) = Iterators.filter(∋(i) ∘ inds, leaves(path)) |> first |> Base.Fix2(size, i)

Base.size(tensor::Tensor) = size(tensor.array)
Base.size(tensor::Tensor, i) = size(tensor.array, i)
Base.size(tensor::Tensor, index::Symbol) = size(tensor, findfirst(==(index), tensor.inds))

"""
    collapse!(path::EinExpr)

Collapses all sub-branches, merging all tensor leaves into the `args` field.
"""
collapse!(path) = path.args = leaves(path) |> collect

"""
    select(path::EinExpr, i)

Return the child elements that contain `i` indices.
"""
select(path::EinExpr, i) = filter(∋(i) ∘ head, args(path))
select(path::EinExpr, i::Base.AbstractVecOrTuple) = ∩(Iterators.map(j -> select(path, j), i)...)

"""
    neighbours(path::EinExpr, i)

Return the indices neighbouring to `i`.
"""
neighbours(path::EinExpr, i) = setdiff(∪(head.(select(path, i))...), (i,))
neighbours(path::EinExpr, i::Base.AbstractVecOrTuple) = setdiff(∪(head.(select(path, i))...), i)

"""
    contractorder(path::EinExpr)

Transform `path` into a contraction order.
"""
contractorder(path::EinExpr) = map(suminds, branches(path))

@doc raw"""
    suminds(path[, parallel=false])

Indices of summation of an `EinExpr`.

```math
\mathtt{path} \equiv \sum_{j k l m n o p} A_{mi} B_{ijp} C_{jkn} D_{pkl} E_{mno} F_{ol}
```

```julia
suminds(path) == [:j, :k, :l, :m, :n, :o, :p]
```
"""
function suminds(path::EinExpr; parallel::Bool = false)
    !parallel && return setdiff(inds(path), head(path)) |> collect

    # annotate connections of indices
    edges = DefaultDict{Symbol,Set{UInt}}(() -> Set{UInt}())
    for input in args(path)
        for index in head(input)
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
    sum!(path, indices)

Explicit, in-place sum over `indices`.

See also: [`sum`](@ref), [`suminds`](@ref).
"""
function Base.sum!(path::EinExpr, inds)
    i = .!isdisjoint.((inds,), head.(args(path)))

    subargs = splice!(args(path), findall(i))
    subinds = head.(subargs)
    subsuminds = setdiff(∩(subinds...), head(path))
    subhead = setdiff(Iterators.flatten(subinds), subsuminds)

    pushfirst!(path.args, EinExpr(subhead, subargs))
    return path
end

"""
    sum(path, indices)

Explicit sum over `indices`.

See [`sum!`](@ref) for inplace modification.
"""
function Base.sum(path::EinExpr, inds::Union{Symbol,AbstractVecOrTuple{Symbol}})
    i = .!isdisjoint.((inds,), head.(args(path)))

    subinds = head.(args(path)[findall(i)])
    subsuminds = setdiff(∩(subinds...), head(path))
    suboutput = setdiff(Iterators.flatten(subinds), subsuminds)

    return EinExpr(head(path), (EinExpr(suboutput, args(path)[findall(i)]), args(path)[findall(.!i)]...))
end

Base.sum(inputs::Union{Tensor,EinExpr}...; inds = mapreduce(head, symdiff, inputs)) = EinExpr(inds, inputs)

function Base.string(path::EinExpr; recursive::Bool = false)
    !recursive && return "$(join(map(x -> string.(head(x)) |> join, args(path)), ","))->$(string.(head(path)) |> join)"
end

# Iteration interface
Base.IteratorSize(::Type{EinExpr}) = Base.HasLength()
Base.length(path::EinExpr) = sum(arg -> arg isa EinExpr ? length(arg) : 1, args(path)) + 1
Base.IteratorEltype(::Type{EinExpr}) = Base.HasEltype()
Base.eltype(::EinExpr) = Union{<:Tensor,EinExpr}

function Base.iterate(path::EinExpr, state = 1)
    isnothing(state) && return nothing

    # iterate child level
    i, j... = state
    it = iterate(args(path), i)

    # return itself on last iteration
    isnothing(it) && return path, nothing

    # recurse iteration
    (next, statenext) = it

    # if `next` is a tensor, return directly
    !(next isa EinExpr) && return next, statenext

    next, j = if isempty(j)
        iterate(next)
    else
        iterate(next, j)
    end

    # if j === nothing, args(path) iteration has finished
    isnothing(j) && return next, i + 1

    return next, (i, j...)
end
