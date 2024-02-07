using AbstractTrees

struct SizedEinExpr{Label}
    path::EinExpr{Label}
    size::Dict{Label,Int}

    function SizedEinExpr(path::EinExpr{L}, size) where {L}
        # inds(path) ⊆ keys(size) || throw(ArgumentError(""))
        new{L}(path, size)
    end
end

EinExpr(path::Vector{L}, size::Dict{L}) where {L} = SizedEinExpr(EinExpr(path), size)

head(sexpr::SizedEinExpr) = head(sexpr.path)

"""
    args(sexpr::SizedEinExpr)

# Note

Unlike `args(::EinExpr)`, this function returns `SizedEinExpr` objects.
"""
args(sexpr::SizedEinExpr) = map(Base.Fix2(SizedEinExpr, sexpr.size), sexpr.path.args) # sexpr.path.args

nargs(sexpr::SizedEinExpr) = nargs(sexpr.path)
inds(sexpr::SizedEinExpr) = inds(sexpr.path)

function Base.getproperty(sexpr::SizedEinExpr, name::Symbol)
    name === :head && return getfield(sexpr, :path).head
    name === :args && return getfield(sexpr, :path).args
    return getfield(sexpr, name)
end

Base.:(==)(a::SizedEinExpr, b::SizedEinExpr) = a.path == b.path && a.size == b.size

Base.ndims(sexpr::SizedEinExpr) = ndims(sexpr.path)

Base.size(sexpr::SizedEinExpr) = size(sexpr.path, sexpr.size)
Base.size(sexpr::SizedEinExpr, i) = sexpr.size[i]
Base.length(sexpr::SizedEinExpr) = length(sexpr.path, sexpr.size)

collapse!(sexpr::SizedEinExpr) = collapse!(sexpr.path)

select(sexpr::SizedEinExpr, i) = map(Base.Fix2(SizedEinExpr, sexpr.size), select(sexpr.path, i))

neighbours(sexpr::SizedEinExpr, i) = map(Base.Fix2(SizedEinExpr, sexpr.size), neighbours(sexpr.path, i))

contractorder(sexpr::SizedEinExpr) = contractorder(sexpr.path)

hyperinds(sexpr::SizedEinExpr) = hyperinds(sexpr.path)

suminds(sexpr::SizedEinExpr) = suminds(sexpr.path)
parsuminds(sexpr::SizedEinExpr) = parsuminds(sexpr.path)

Base.sum!(sexpr::SizedEinExpr, inds) = sum!(sexpr.path, inds)
Base.sum(sexpr::SizedEinExpr, inds) = sum(sexpr.path, inds)

function Base.sum(sexpr::Vector{SizedEinExpr{L}}; skip = L[]) where {L}
    path = sum(map(x -> x.path, sexpr); skip)
    size = allequal(Iterators.map(x -> x.size, sexpr)) ? first(sexpr).size : merge(map(x -> x.size, sexpr)...)
    # size = merge(map(x -> x.size, sexpr)...)
    SizedEinExpr(path, size)
end

# Iteration interface
Base.IteratorEltype(::Type{<:TreeIterator{SizedEinExpr}}) = Base.HasEltype()
Base.eltype(::Type{<:TreeIterator{SizedEinExpr}}) = SizedEinExpr

# AbstractTrees interface and traits
AbstractTrees.children(sexpr::SizedEinExpr) = args(sexpr)
AbstractTrees.childtype(::Type{SizedEinExpr}) = SizedEinExpr
AbstractTrees.childrentype(::Type{SizedEinExpr}) = Vector{SizedEinExpr}
AbstractTrees.childstatetype(::Type{SizedEinExpr}) = Int
AbstractTrees.nodetype(::Type{SizedEinExpr}) = SizedEinExpr

AbstractTrees.ParentLinks(::Type{SizedEinExpr}) = ImplicitParents()
AbstractTrees.SiblingLinks(::Type{SizedEinExpr}) = ImplicitSiblings()
AbstractTrees.ChildIndexing(::Type{SizedEinExpr}) = IndexedChildren()
AbstractTrees.NodeType(::Type{SizedEinExpr}) = HasNodeType()

# Utils
sumtraces(path::SizedEinExpr) = SizedEinExpr(sumtraces(path.path), path.size)
