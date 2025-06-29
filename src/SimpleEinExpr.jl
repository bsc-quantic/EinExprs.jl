using Base: AbstractVecOrTuple
using DataStructures: DefaultDict
using AbstractTrees
using Compat

Base.@kwdef struct SimpleEinExpr{Label}
    head::Vector{Label}
    args::Vector{SimpleEinExpr{Label}} = SimpleEinExpr{Label}[]
end

SimpleEinExpr(head::Vector{L}) where {L} = SimpleEinExpr(head, SimpleEinExpr{L}[])
SimpleEinExpr(head::Vector{L}, args::Vector{Vector{L}}) where {L} = SimpleEinExpr{L}(head, map(SimpleEinExpr{L}, args))

SimpleEinExpr(head::NTuple, args) = SimpleEinExpr(collect(head), args)
SimpleEinExpr(head, args::NTuple) = SimpleEinExpr(head, collect(args))
SimpleEinExpr(head::NTuple, args::NTuple) = SimpleEinExpr(collect(head), collect(args))

Base.copy(expr::SimpleEinExpr) = SimpleEinExpr(copy(expr.head), copy(expr.args))
Base.:(==)(a::EinExpr, b::EinExpr) = a.head == b.head && a.args == b.args

Base.show(io::IO, path::SimpleEinExpr) = print_tree((io, node) -> print(io, head(node)), io, path)
function Base.string(path::EinExpr; recursive::Bool = false)
    !recursive && return "$(join(map(x -> string.(head(x)) |> join, args(path)), ","))->$(string.(head(path)) |> join)"
    map(string, Branches(path))
end

head(path::SimpleEinExpr) = path.head
args(path::SimpleEinExpr) = path.args
nargs(path::SimpleEinExpr) = length(path.args)

Base.size(path::EinExpr, sizedict) = (sizedict[i] for i in head(path)) |> @compat(splat(tuple))
Base.length(path::EinExpr, sizedict) = (prod ∘ size)(path, sizedict)

"""
    collapse!(path::EinExpr)

Collapses all sub-branches, merging all tensor leaves into the `args` field.
"""
collapse!(path) = path.args = leaves(path)

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

# TODO may need a fix for contracting hyperindices, or other ege cases
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

#indshistogram(exprs::Vector) = indshistogram(exprs...)

function indshistogram(exprs::Vector{EinExpr{L}}) where {L}
    histogram = Dict{L,Int}()

    for expr in exprs, i in head(expr)
        count = get!(histogram, i, 0)
        histogram[i] = count + 1
    end

    return histogram
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
