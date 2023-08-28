using Base: AbstractVecOrTuple
using DataStructures: DefaultDict
using ImmutableArrays
using AbstractTrees

struct EinExpr
    head::ImmutableVector{Symbol,Vector{Symbol}}
    args::Vector{EinExpr}
    size::Dict{Symbol,Int}

    function EinExpr(head, args)
        # TODO checks: same dim for index, valid indices
        head = collect(head)
        args = collect(args)
        new(head, args, Dict{Symbol,EinExpr}())
    end

    function EinExpr(head, size::AbstractDict{Symbol,Int})
        issetequal(head, keys(size)) || throw(ArgumentError("Missing sizes for indices $(setdiff(head, keys(size)))"))
        new(head, EinExpr[], size)
    end
end

"""
    head(path::EinExpr)

Return the indices of the resulting tensor from contracting `path`.

See also: [`inds`](@ref), [`args`](@ref).
"""
head(path::EinExpr) = path.head

"""
    args(path::EinExpr)

Return the children of the `path`, which correspond to input tensors for the contraction step in the top of the `path`.

See also: [`head`](@ref).
"""
args(path::EinExpr) = path.args

"""
    inds(path)

Return all the involved indices in `path`. If a tensor is passed, then it is equivalent to calling [`head`](@ref).

See also: [`head`](@ref).
"""
inds(path::EinExpr) = mapreduce(parent ∘ head, ∪, Leaves(path)) |> collect

"""
    leaves(path::EinExpr[, i])

Return the terminal leaves of the `path`, which correspond to the initial input tensors.
If `i` is specified, then only return the ``i``-th tensor.

See also: [`branches`](@ref).
"""
leaves(path) = Leaves(path) |> collect
leaves(path, i) = Iterators.drop(Leaves(path), i - 1) |> first

Branches(path) = Iterators.filter(!isempty ∘ args, PostOrderDFS(path))

"""
    branches(path::EinExpr[, i])

Return the non-terminal branches of the `path`, which correspond to intermediate tensors result of contraction steps.
If `i` is specified, then only return the ``i``-th `EinExpr`.

See also: [`leaves`](@ref).
"""
branches(path) = Branches(path) |> collect
branches(path, i) = Iterators.drop(Branches(path), i - 1) |> first

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
Base.size(path::EinExpr) = (size(path, i) for i in head(path)) |> splat(tuple)
Base.size(path::EinExpr, i::Symbol) =
    Iterators.filter(∋(i) ∘ head, Leaves(path)) |> first |> Base.Fix2(getproperty, :size) |> Base.Fix2(getindex, i)

Base.length(path::EinExpr) = (prod ∘ size)(path)

"""
    collapse!(path::EinExpr)

Collapses all sub-branches, merging all tensor leaves into the `args` field.
"""
collapse!(path) = path.args = leaves(path)

"""
    select(path::EinExpr, i)

Return the child elements that contain `i` indices.
"""
select(path::EinExpr, i) = filter(∋(i) ∘ head, args(path))
select(path::EinExpr, i::Base.AbstractVecOrTuple) = filter(Base.Fix1(⊆, collect(i)) ∘ head, args(path))

"""
    neighbours(path::EinExpr, i)

Return the indices neighbouring to `i`.
"""
neighbours(path::EinExpr, i) = neighbours(path, (i,))
neighbours(path::EinExpr, i::Base.AbstractVecOrTuple) = setdiff(mapreduce(head, ∪, select(path, i), init = Symbol[]), i)

"""
    contractorder(path::EinExpr)

Transform `path` into a contraction order.
"""
contractorder(path::EinExpr) = map(suminds, Branches(path))

@doc raw"""
    suminds(path)

Indices of summation of an `EinExpr`.

```math
\mathtt{path} \equiv \sum_{j k l m n o p} A_{mi} B_{ijp} C_{jkn} D_{pkl} E_{mno} F_{ol}
```

```julia
suminds(path) == [:j, :k, :l, :m, :n, :o, :p]
```
"""
suminds(path::EinExpr) = setdiff(mapreduce(head, ∪, path.args), head(path))

# TODO keep output inds
parsuminds(path::EinExpr) =
    Iterators.filter(!isempty, Iterators.map(((a, b),) -> suminds(sum([a, b])), combinations(path.args, 2))) |> collect

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

Base.sum(args::Vector{EinExpr}; head = mapreduce(head, symdiff, args)) = EinExpr(head, args)

function Base.string(path::EinExpr; recursive::Bool = false)
    !recursive && return "$(join(map(x -> string.(head(x)) |> join, args(path)), ","))->$(string.(head(path)) |> join)"
end

# Iteration interface
Base.IteratorEltype(::Type{<:TreeIterator{EinExpr}}) = Base.HasEltype()
Base.eltype(::Type{<:TreeIterator{EinExpr}}) = EinExpr

# AbstractTrees interface and traits
AbstractTrees.children(path::EinExpr) = args(path)
AbstractTrees.childtype(::Type{EinExpr}) = EinExpr
AbstractTrees.childrentype(::Type{EinExpr}) = Vector{EinExpr}
AbstractTrees.childstatetype(::Type{EinExpr}) = Int
AbstractTrees.nodetype(::Type{EinExpr}) = EinExpr

AbstractTrees.ParentLinks(::Type{EinExpr}) = ImplicitParents()
AbstractTrees.SiblingLinks(::Type{EinExpr}) = ImplicitSiblings()
AbstractTrees.ChildIndexing(::Type{EinExpr}) = IndexedChildren()
AbstractTrees.NodeType(::Type{EinExpr}) = HasNodeType()
