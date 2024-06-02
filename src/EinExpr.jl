using Base: AbstractVecOrTuple
using DataStructures: DefaultDict
using AbstractTrees
using Compat

Base.@kwdef struct EinExpr{Label}
    head::Vector{Label}
    args::Vector{EinExpr{Label}} = EinExpr{Label}[]
end

EinExpr(head::Vector{L}) where {L} = EinExpr(head, EinExpr{L}[])
EinExpr(head::Vector{L}, args::Vector{Vector{L}}) where {L} = EinExpr{L}(head, map(EinExpr{L}, args))

EinExpr(head::NTuple, args) = EinExpr(collect(head), args)
EinExpr(head, args::NTuple) = EinExpr(head, collect(args))
EinExpr(head::NTuple, args::NTuple) = EinExpr(collect(head), collect(args))

Base.copy(expr::EinExpr) = EinExpr(copy(expr.head), copy(expr.args))

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

nargs(path::EinExpr) = length(path.args)

"""
    inds(path)

Return all the involved indices in `path`. If a tensor is passed, then it is equivalent to calling [`head`](@ref).

See also: [`head`](@ref).
"""
inds(path::EinExpr{L}) where {L} = mapreduce(head, ∪, Leaves(path); init = L[])

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
Base.size(path::EinExpr, sizedict) = (sizedict[i] for i in head(path)) |> @compat(splat(tuple))
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
neighbours(path::EinExpr{L}, i::Base.AbstractVecOrTuple) where {L} =
    setdiff(mapreduce(head, union!, Iterators.filter(node -> i ⊆ head(node), Leaves(path)), init = L[]), i)

"""
    contractorder(path::EinExpr)

Transform `path` into a contraction order.
"""
contractorder(path::EinExpr) = map(suminds, Branches(path))

hyperinds(path::EinExpr) =
    map(
        first,
        Iterators.filter(
            >(2) ∘ last,
            Iterators.map(i -> (i, count(∋(i) ∘ head, args(path))), Iterators.flatten(Iterators.map(head, args(path)))),
        ),
    ) |> unique!

openinds(path::EinExpr) =
    map(
        first,
        Iterators.filter(
            ==(1) ∘ last,
            Iterators.map(i -> (i, count(∋(i) ∘ head, args(path))), Iterators.flatten(Iterators.map(head, args(path)))),
        ),
    ) |> unique!

# TODO may need a fix for contracting hyperindices, or other edge cases
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

indshistogram(exprs...) = mergewith(+, map(exprs) do expr
    Dict(i => 1 for i in head(expr))
end...)
indshistogram(exprs::Vector) = indshistogram(exprs...)

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
function Base.sum(path::EinExpr{L}, inds::AbstractVecOrTuple{L}) where {L}
    i = .!isdisjoint.((inds,), head.(args(path)))

    subinds = head.(args(path)[findall(i)])
    subsuminds = setdiff(∩(subinds...), head(path))
    suboutput = setdiff(Iterators.flatten(subinds), subsuminds)

    return EinExpr(head(path), (EinExpr(suboutput, args(path)[findall(i)]), args(path)[findall(.!i)]...))
end
Base.sum(path::EinExpr{L}, inds::L) where {L} = sum(path, (inds,))

"""
    sum(tensors::Vector{EinExpr}; skip = [])

Create an `EinExpr` from other `EinExpr`s.

# Keyword arguments

  - `skip` Specifies indices to be skipped from summation.
"""
function Base.sum(args::Vector{EinExpr{L}}; skip = L[]) where {L}
    _head = L[]
    _counts = Int[]

    for arg in args
        for index in head(arg)
            i = findfirst(Base.Fix1(===, index), _head)
            if isnothing(i)
                push!(_head, index)
                push!(_counts, 1)
            else
                @inbounds _counts[i] += 1
            end
        end
    end

    # NOTE `map` with `Iterators.filter` induces many heap grows; allocating once and deleting is faster
    for i in Iterators.reverse(eachindex(_head, _counts))
        (_counts[i] == 1 || _head[i] ∈ skip) && continue
        deleteat!(_head, i)
    end

    EinExpr(_head, args)
end

function Base.sum(a::EinExpr{L}, b::EinExpr{L}; skip = L[]) where {L}
    _head = copy(head(a))

    for index in head(b)
        i = findfirst(Base.Fix1(===, index), _head)
        if isnothing(i)
            push!(_head, index)
        elseif index ∈ skip
            continue
        else
            deleteat!(_head, i)
        end
    end

    EinExpr(_head, [a, b])
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
AbstractTrees.childtype(::Type{E}) where {E<:EinExpr} = E
AbstractTrees.childrentype(::Type{E}) where {E<:EinExpr} = Vector{E}
AbstractTrees.childstatetype(::Type{<:EinExpr}) = Int
AbstractTrees.nodetype(::Type{E}) where {E<:EinExpr} = E

AbstractTrees.ParentLinks(::Type{<:EinExpr}) = ImplicitParents()
AbstractTrees.SiblingLinks(::Type{<:EinExpr}) = ImplicitSiblings()
AbstractTrees.ChildIndexing(::Type{<:EinExpr}) = IndexedChildren()
AbstractTrees.NodeType(::Type{<:EinExpr}) = HasNodeType()

# Utils
function sumtraces(path::EinExpr)
    do_not_contract_inds = hyperinds(path) ∪ path.head
    _args = map(path.args) do arg
        selfinds = nonunique(arg.head)
        isempty(selfinds) && return arg

        skip = selfinds ∩ do_not_contract_inds
        sum([arg]; skip)
    end

    EinExpr(path.head, _args)
end
