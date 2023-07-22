Base.selectdim(path::EinExpr, index::Symbol, i) = EinExpr(map(path.args) do sub
    index ∈ __labels_children(sub) ? selectdim(sub, index, i) : sub
end, filter(!=(index), path.head))

__labels_children(x) = labels(x)
__labels_children(path::EinExpr) = labels(path, all = true)

Base.view(path::EinExpr, cuttings::Pair{Symbol,<:Integer}...) =
    reduce(cuttings, init = path) do acc, proj
        d, i = proj
        selectdim(acc, d, i)
    end

function findslices(
    scorer,
    path::EinExpr;
    size = nothing,
    overhead = nothing,
    slices = nothing,
    temperature = 0.01,
    skip = labels(path),
)
    all(isnothing, (size, overhead, slices)) &&
        throw(ArgumentError("need to specify at least one size, overhead or slices target"))

    candidates = Set(setdiff(mapreduce(labels, ∪, path), skip))
    solution = Set{Symbol}()
    current = (; slices = 1, size = maximum(prod ∘ Base.size, path), overhead = 1.0)
    original_flops = mapreduce(flops, +, path)

    sliced_path = path
    while !isempty(candidates)
        # temperature adds boltzmann like noise
        winner = argmax(candidates) do index
            scorer(sliced_path, index) - temperature * (log ∘ (-) ∘ log ∘ rand)()
        end
        delete!(candidates, winner)

        sliced_path = selectdim(sliced_path, winner, 1)
        cur_overhead =
            prod(i -> Base.size(path, i), [solution..., winner]) * mapreduce(flops, +, sliced_path) / original_flops

        !isnothing(overhead) && cur_overhead > overhead && break
        push!(solution, winner)

        current = (;
            slices = current.slices * (prod ∘ Base.size)(path, winner),
            size = maximum(prod ∘ Base.size, sliced_path),
            overhead = cur_overhead,
        )

        !isnothing(slices) && current.slices >= slices && break
        !isnothing(size) && current.size <= size && break
    end

    return solution
end

abstract type Scorer end

Base.@kwdef struct FlopsScorer <: Scorer
    weight::Float64 = 1e-3
end

function (cb::FlopsScorer)(path, index)
    slice = selectdim(path, index, 1)

    flops_reduction = mapreduce(flops, +, path) - mapreduce(flops, +, slice)
    write_reduction = mapreduce(prod ∘ size, +, path) - mapreduce(prod ∘ size, +, slice)

    log(flops_reduction + write_reduction * cb.weight + 1)
end

Base.@kwdef struct SizeScorer <: Scorer
    weight::Float64 = 1e-3
end

function (cb::SizeScorer)(path, index)
    slice = selectdim(path, index, 1)

    flops_reduction = mapreduce(flops, +, path) - mapreduce(flops, +, slice)
    write_reduction = mapreduce(prod ∘ size, +, path) - mapreduce(prod ∘ size, +, slice)

    log(write_reduction + flops_reduction * cb.weight + 1)
end
