using DelegatorTraits
using AbstractTrees

abstract type AbstractEinExpr{L} end

"""
    EinExpr <: Interface

Interface object of a `EinExpr`; i.e. the contraction path interface.

Aside of the methods intrinsic to this package, a type implementing `EinExpr` needs also to implement the `AbstractTree`
interface (from AbstractTrees.jl).
"""
struct EinExpr <: Interface end

# required interface methods
"""
    head(path)

Return the indices of the resulting tensor from contracting `path`.

See also: [`inds`](@ref), [`args`](@ref).
"""
function head end

"""
    args(path)

Return the children of `path`, which correspond to input tensors for the contraction step in the top of `path`.

See also: [`head`](@ref).
"""
function args end

"""
    Base.size(path[, index])

Return the size of the resulting tensor from contracting `path`. If `index` is specified, return the size of such index.
"""
:(Base.size(::AbstractEinExpr, args...))

# TODO mutating interface methods

# optional methods
"""
    nargs(path)

Return the number of children of `path`.

See also: [`args`](@ref).
"""
function nargs end

"""
    inds(path)

Return all the involved indices in `path`. If a leaf is passed, then it is equivalent to calling [`head`](@ref).

See also: [`head`](@ref).
"""
function inds end

"""
    Base.ndims(path)

Return the number of indices of the resulting tensor from contracting `path`.
"""
:(Base.ndims(::AbstractEinExpr))

# needs to implement `AbstractTree` interface
:(Base.IteratorEltype(::Type{<:TreeIterator{AbstractEinExpr}}))
:(Base.eltype(::Type{<:TreeIterator{AbstractEinExpr}}))
:(AbstractTrees.children(path::AbstractEinExpr))
:(AbstractTrees.childtype(::Type{E}) where {E<:AbstractEinExpr})
:(AbstractTrees.childrentype(::Type{E}) where {E<:AbstractEinExpr})
:(AbstractTrees.childstatetype(::Type{<:AbstractEinExpr}))
:(AbstractTrees.nodetype(::Type{E}) where {E<:AbstractEinExpr})
:(AbstractTrees.ParentLinks(::Type{<:AbstractEinExpr}))
:(AbstractTrees.SiblingLinks(::Type{<:AbstractEinExpr}))
:(AbstractTrees.ChildIndexing(::Type{<:AbstractEinExpr}))
:(AbstractTrees.NodeType(::Type{<:AbstractEinExpr}))

# non-delegated methods
"""
    select(path::EinExpr, i)

Return the child elements that contain `i` indices.
"""
function select end

"""
    neighbours(path::EinExpr, i)

Return the indices neighbouring to `i`.
"""
function neighbors end

"""
    leaves(path::EinExpr[, i])

Return the terminal leaves of the `path`, which correspond to the initial input tensors.
If `i` is specified, then only return the ``i``-th tensor.

See also: [`branches`](@ref).
"""
function leaves end

"""
    Branches(path::EinExpr)

Iterator that walks through the non-terminal nodes of the `path` tree.

See also: [`branches`](@ref).
"""
function Branches end

"""
    branches(path::EinExpr[, i])

Return the non-terminal branches of the `path`, which correspond to intermediate tensors result of contraction steps.
If `i` is specified, then only return the ``i``-th `EinExpr`.

See also: [`leaves`](@ref), [`Branches`](@ref).
"""
function branches end

# implementation
## `head`
head(x) = head(x, DelegatorTrait(EinExpr, x))
head(x, ::DelegateToField) = head(delegator(EinExpr, x))
head(x, ::DontDelegate) = throw(MethodError(head, (x,)))

## `args`
args(x) = args(x, DelegatorTrait(EinExpr, x))
args(x, ::DelegateToField) = args(delegator(EinExpr, x))
args(x, ::DontDelegate) = throw(MethodError(args, (x,)))

## `Base.size`
Base.size(x::AbstractEinExpr) = size(x, DelegatorTrait(EinExpr, x))
Base.size(x::AbstractEinExpr, ::DelegateToField) = size(delegator(EinExpr, x))
Base.size(x::AbstractEinExpr) = throw(MethodError(size, (x)))

Base.size(x::AbstractEinExpr, ind) = size(x, ind, DelegatorTrait(EinExpr, x))
Base.size(x::AbstractEinExpr, ind, ::DelegateToField) = size(delegator(EinExpr, x), ind)
Base.size(x::AbstractEinExpr, ind) = throw(MethodError(size, (x, ind)))

## `nargs`
nargs(x) = nargs(x, DelegatorTrait(EinExpr, x))
nargs(x, ::DelegateToField) = args(delegator(EinExpr, x))
function nargs(x, ::DontDelegate)
    fallback(args)
    return length(args(x))
end

## `inds`
inds(x::AbstractEinExpr) = inds(x, DelegatorTrait(EinExpr, x))
inds(x::AbstractEinExpr, ::DelegateToField) = inds(delegator(EinExpr, x))
function inds(x::AbstractEinExpr{L}, ::DontDelegate) where {L}
    fallback(inds)
    return mapreduce(head, ∪, Leaves(path); init = L[])
end

## `Base.ndims`
Base.ndims(x::AbstractEinExpr) = ndims(x, DelegatorTrait(EinExpr, x))
Base.ndims(x::AbstractEinExpr, ::DelegateToField) = ndims(delegator(EinExpr, x))
function Base.ndims(x::AbstractEinExpr)
    fallback(ndims)
    return length(head(path))
end

## `AbstractTree` interface
Base.IteratorEltype(::Type{<:TreeIterator{<:AbstractEinExpr}}) = Base.HasEltype()
Base.eltype(::Type{<:TreeIterator{E}}) where {E<:AbstractEinExpr} = E

AbstractTrees.children(path::AbstractEinExpr) = args(path)
AbstractTrees.childtype(::Type{E}) where {E<:AbstractEinExpr} = E
AbstractTrees.childrentype(::Type{E}) where {E<:AbstractEinExpr} = Vector{E}
AbstractTrees.childstatetype(::Type{<:AbstractEinExpr}) = Int
AbstractTrees.nodetype(::Type{E}) where {E<:AbstractEinExpr} = E

AbstractTrees.ParentLinks(::Type{<:AbstractEinExpr}) = ImplicitParents()
AbstractTrees.SiblingLinks(::Type{<:AbstractEinExpr}) = ImplicitSiblings()
AbstractTrees.ChildIndexing(::Type{<:AbstractEinExpr}) = IndexedChildren()
AbstractTrees.NodeType(::Type{<:AbstractEinExpr}) = HasNodeType()

## `select`
select(path, i) = Iterators.filter(∋(i) ∘ head, PreOrderDFS(path)) |> collect
select(path, i::AbstractVecOrTuple) = Iterators.filter(Base.Fix1(⊆, i) ∘ head, PreOrderDFS(path)) |> collect

## `neighbors`
neighbors(path, i) = neighbors(path, (i,))
function neighbors(path, i::Base.AbstractVecOrTuple)
    setdiff(mapreduce(head, union!, Iterators.filter(node -> i ⊆ head(node), Leaves(path)), init = L[]), i)
end

## `leaves`
leaves(path) = Leaves(path) |> collect
leaves(path, i) = Iterators.drop(Leaves(path), i - 1) |> first

## `Branches`
Branches(path; inverse = false) = Iterators.filter(!isempty ∘ args, (inverse ? PreOrderDFS : PostOrderDFS)(path))

## `branches`
branches(path; inverse = false) = Branches(path; inverse) |> collect
branches(path, i; inverse = false) = Iterators.drop(Branches(path; inverse), i - 1) |> first
