using Base: AbstractVecOrTuple
using DataStructures: DefaultDict
using AbstractTrees

Base.@kwdef struct EinExpr
    head::Vector{Symbol}
    args::Vector{EinExpr} = EinExpr[]
end

EinExpr(head) = EinExpr(head, EinExpr[])
EinExpr(head, args::AbstractVecOrTuple{<:AbstractVecOrTuple{Symbol}}) = EinExpr(head, map(EinExpr, args))

EinExpr(head::NTuple, args) = EinExpr(collect(head), args)
EinExpr(head, args::NTuple) = EinExpr(head, collect(args))
EinExpr(head::NTuple, args::NTuple) = EinExpr(collect(head), collect(args))

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

"""
    Branches(path::EinExpr)

Iterator that walks through the non-terminal nodes of the `path` tree.

See also: [`branches`](@ref).
"""
Branches(path; inverse = false) = Iterators.filter(!isempty ∘ args, (inverse ? PreOrderDFS : PostOrderDFS)(path))

"""
    branches(path::EinExpr[, i])

Return the non-terminal branches of the `path`, which correspond to intermediate tensors result of contraction steps.
If `i` is specified, then only return the ``i``-th `EinExpr`.

See also: [`leaves`](@ref), [`Branches`](@ref).
"""
branches(path; inverse = false) = Branches(path; inverse) |> collect
branches(path, i; inverse = false) = Iterators.drop(Branches(path; inverse), i - 1) |> first

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
Base.size(path::EinExpr, sizedict) = (sizedict[i] for i in head(path)) |> splat(tuple)
Base.length(path::EinExpr, sizedict) = (prod ∘ size)(path, sizedict)

"""
    collapse!(path::EinExpr)

Collapses all sub-branches, merging all tensor leaves into the `args` field.
"""
collapse!(path) = path.args = leaves(path)

"""
    select(path::EinExpr, i)

Return the child elements that contain `i` indices.
"""
select(path::EinExpr, i) = Iterators.filter(∋(i) ∘ head, PreOrderDFS(path)) |> collect
select(path::EinExpr, i::AbstractVecOrTuple) = Iterators.filter(Base.Fix1(⊆, i) ∘ head, PreOrderDFS(path)) |> collect

"""
    neighbours(path::EinExpr, i)

Return the indices neighbouring to `i`.
"""
neighbours(path::EinExpr, i) = neighbours(path, (i,))
neighbours(path::EinExpr, i::Base.AbstractVecOrTuple) =
    setdiff(mapreduce(head, union!, Iterators.filter(node -> i ⊆ head(node), Leaves(path)), init = Symbol[]), i)

"""
    contractorder(path::EinExpr)

Transform `path` into a contraction order.
"""
contractorder(path::EinExpr) = map(suminds, Branches(path))

hyperinds(path::EinExpr) = map(
    first,
    Iterators.filter(
        >(2) ∘ last,
        Iterators.map(i -> (i, count(∋(i) ∘ head, args(path))), Iterators.flatmap(head, args(path))),
    ),
)

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
suminds(path::EinExpr) = filter!(∉(head(path)), flatunique(head, path.args))

@generated function flatunique(f, itr)
    if Iterators.IteratorEltype(itr) isa Iterators.EltypeUnknown
        return :(flatunique(Any, f, itr))
    end

    fouttype = Base.promote_op(f.instance, eltype(itr))
    if Iterators.IteratorEltype(fouttype) isa Iterators.EltypeUnknown
        return :(flatunique(Any, f, itr))
    end

    return :(flatunique($(eltype(fouttype)), f, itr))
end

function flatunique(::Type{T}, f, itr) where {T}
    u = T[]
    for x in itr
        for y in f(x)
            y ∉ u && push!(u, y)
        end
    end

    return u
end

# TODO keep output inds
"""
    parsuminds(path)

Indices of summation of possible pairwise tensors contractions between children of `path`.
"""
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

"""
    sum(tensors::Vector{EinExpr}; skip = Symbol[])

Create an `EinExpr` from other `EinExpr`s.

# Keyword arguments

  - `skip` Specifies indices to be skipped from summation.
"""
function Base.sum(args::Vector{EinExpr}; skip = Symbol[])
    _head = Symbol[]
    _counts = Int[]
    for arg in args
        for index in head(arg)
            i = findfirst(Base.Fix1(===, index), _head)
            if isnothing(i)
                push!(_head, index)
                push!(_counts, 1)
            else
                _counts[i] += 1
            end
        end
    end

    _head = map(first, Iterators.filter(zip(_head, _counts)) do (index, count)
        count == 1 || index ∈ skip
    end)
    EinExpr(_head, args)
end

function Base.string(path::EinExpr; recursive::Bool = false)
    !recursive && return "$(join(map(x -> string.(head(x)) |> join, args(path)), ","))->$(string.(head(path)) |> join)"
    map(string, Branches(path))
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
