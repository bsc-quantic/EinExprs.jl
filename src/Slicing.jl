using AbstractTrees

"""
    selectdim(path::EinExpr, index::Symbol, i)

Project `index` to dimension `i` in a EinExpr. This is equivalent to tensor cutting aka slicing.

# Arguments

  - `path` Contraction path.
  - `index` Index to cut.
  - `i` Dimension of `index` to select.

See also: [`view`](@ref).
"""
Base.selectdim(path::EinExpr, index::Symbol, i) = EinExpr(filter(!=(index), head(path)), map(args(path)) do sub
    index ∈ __inds_children(sub) ? selectdim(sub, index, i) : sub
end)

__inds_children(x) = head(x)
__inds_children(path::EinExpr) = inds(path)

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
Base.view(path::EinExpr, cuttings::Pair{Symbol,<:Integer}...) =
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
    path::EinExpr;
    size = nothing,
    overhead = nothing,
    slices = nothing,
    temperature = 0.01,
    skip = head(path),
)
    all(isnothing, (size, overhead, slices)) &&
        throw(ArgumentError("need to specify at least one size, overhead or slices target"))

    candidates = Set(setdiff(mapreduce(head, ∪, PostOrderDFS(path)), skip))
    solution = Set{Symbol}()
    current = (; slices = 1, size = maximum(prod ∘ Base.size, PostOrderDFS(path)), overhead = 1.0)
    original_flops = mapreduce(flops, +, Branches(path))

    sliced_path = path
    while !isempty(candidates)
        # temperature adds boltzmann like noise
        winner = argmax(candidates) do index
            scorer(sliced_path, index) - temperature * (log ∘ (-) ∘ log ∘ rand)()
        end
        delete!(candidates, winner)

        sliced_path = selectdim(sliced_path, winner, 1)
        cur_overhead =
            prod(i -> Base.size(path, i), [solution..., winner]) * mapreduce(flops, +, Branches(sliced_path)) /
            original_flops

        !isnothing(overhead) && cur_overhead > overhead && break
        push!(solution, winner)

        current = (;
            slices = current.slices * (prod ∘ Base.size)(path, winner),
            size = maximum(prod ∘ Base.size, PostOrderDFS(sliced_path)),
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
    write_reduction = mapreduce(prod ∘ size, +, PostOrderDFS(path)) - mapreduce(prod ∘ size, +, PostOrderDFS(slice))

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
    write_reduction = mapreduce(prod ∘ size, +, PostOrderDFS(path)) - mapreduce(prod ∘ size, +, PostOrderDFS(slice))

    log(write_reduction + flops_reduction * cb.weight + 1)
end
