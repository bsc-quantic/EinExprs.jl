using AbstractTrees
using Compat

"""
    selectdim(path::EinExpr, index, i)

Project `index` to dimension `i` in a EinExpr. This is equivalent to tensor cutting aka slicing.

# Arguments

  - `path` Contraction path.
  - `index` Index to cut.
  - `i` Dimension of `index` to select.

See also: [`view`](@ref).
"""
Base.selectdim(path::EinExpr{L}, ::L, i) where {L} = path

function Base.selectdim(path::EinExpr{L}, index::L, i::Integer) where {L}
    path = deepcopy(path)

    for expr in PreOrderDFS(path)
        filter!(!=(index), expr.head)
    end

    return path
end

function Base.selectdim(sexpr::SizedEinExpr, index, i)
    path = selectdim(sexpr.path, index, i)

    size = copy(sexpr.size)
    size[index] = length(i)

    return SizedEinExpr(path, size)
end

function Base.selectdim(sexpr::SizedEinExpr, index, i::Integer)
    path = selectdim(sexpr.path, index, i)

    size = filter(!=(index) ∘ first, sexpr.size)

    return SizedEinExpr(path, size)
end

"""
    view(path::EinExpr, cuttings...)

Project indices in contraction `path` to some of its dimensions. This is equivalent to:

```julia
reduce(cuttings) do path, (index, i)
    selectdim(path, index, i)
end
```

# Arguments

  - `path` Target contraction path.
  - `cuttings` List of `Pair{Symbol,Int}` representing the tensor cuttings aka slices.

See also: [`selectdim`](@ref).
"""
Base.view(path::EinExpr{L}, cuttings::Pair{L,<:Integer}...) where {L} =
    reduce(cuttings, init = path) do acc, proj
        d, i = proj
        selectdim(acc, d, i)
    end

"""
    findslices(scorer, path::EinExpr; size, slices, overhead, temperature = 0.01, skip = head(path))

Search for indices to be cut/sliced such that the conditions given by `size`, `overhead` and `slices` are fulfilled.
Reimplementation based on [`contengra`](https://github.com/jcmgray/cotengra)'s `SliceFinder` algorithm.

# Arguments

  - `scorer` Heuristic function (or functor) that accepts a path and a candidate index for cutting, and returns a score.
  - `path` The contraction path target for tensor cutting aka slicing.

# Keyword Arguments

  - `size` If specified, the largest intermediate tensor of the slice won't surpass this size (in number of elements).
  - `slices` If specified, there will be at least `slices` different slices when cutting all returnt indices.
  - `overhead` If specified, the amount of redundant operations between a slice and the original contraction won't supass this ratio.
  - `temperature` Temperature of the Boltzmann-like noise added for diffusing results.
  - `skip` Indices not to be considered for slicing.
"""
function findslices(
    scorer,
    path::SizedEinExpr{L};
    size = nothing,
    overhead = nothing,
    slices = nothing,
    temperature = 0.01,
    skip = head(path),
) where {L}
    all(isnothing, (size, overhead, slices)) &&
        throw(ArgumentError("need to specify at least one size, overhead or slices target"))

    candidates = Set(setdiff(mapreduce(head, ∪, PostOrderDFS(path)), skip))
    solution = Set{L}()
    current = @compat (; slices = 1, size = maximum(length, PostOrderDFS(path)), overhead = 1.0)
    original_flops = mapreduce(flops, +, Branches(path; inverse = true))

    sliced_path = path
    while !isempty(candidates)
        # temperature adds boltzmann like noise
        winner = argmax(candidates) do index
            scorer(sliced_path, index) - temperature * (log ∘ (-) ∘ log ∘ rand)()
        end
        delete!(candidates, winner)

        sliced_path = selectdim(sliced_path, winner, 1)
        cur_overhead =
            prod(i -> Base.size(path, i), [solution..., winner]) *
            mapreduce(flops, +, Branches(sliced_path; inverse = true)) / original_flops

        !isnothing(overhead) && cur_overhead > overhead && break
        push!(solution, winner)

        current = @compat (;
            slices = current.slices * Base.size(path, winner),
            size = maximum(length, PostOrderDFS(sliced_path)),
            overhead = cur_overhead,
        )

        !isnothing(slices) && current.slices >= slices && break
        !isnothing(size) && current.size <= size && break
    end

    return solution
end

abstract type Scorer end

"""
    FlopsScorer

# Keyword Arguments

  - `weight`
"""
Base.@kwdef struct FlopsScorer <: Scorer
    weight::Float64 = 1e-3
end

function (cb::FlopsScorer)(path, index)
    slice = selectdim(path, index, 1)

    flops_reduction = mapreduce(flops, +, PostOrderDFS(path)) - mapreduce(flops, +, PostOrderDFS(slice))
    write_reduction = mapreduce(length, +, PostOrderDFS(path)) - mapreduce(length, +, PostOrderDFS(slice))

    log(flops_reduction + write_reduction * cb.weight + 1)
end

"""
    SizeScorer

# Keyword Arguments

  - `weight`
"""
Base.@kwdef struct SizeScorer <: Scorer
    weight::Float64 = 1e-3
end

function (cb::SizeScorer)(path, index)
    slice = selectdim(path, index, 1)

    flops_reduction = mapreduce(flops, +, PostOrderDFS(path)) - mapreduce(flops, +, PostOrderDFS(slice))
    write_reduction = mapreduce(length, +, PostOrderDFS(path)) - mapreduce(length, +, PostOrderDFS(slice))

    log(write_reduction + flops_reduction * cb.weight + 1)
end
