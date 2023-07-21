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

function slices(
    scorer,
    path::EinExpr;
    size = nothing,
    overhead = nothing,
    slices = nothing,
    temperature = 0.01,
    skip = Set{Symbol}(),
)
    candidates = setdiff(labels(path, all = true), skip)
    solution = Set{Symbol}()
    current = (; slices = 1, size = maximum(size, path), overhead = 1.0)
    original_flops = flops(path)

    sliced_path = path
    while !(!isnothing(slices) && current.slices >= slices || !isnothing(size) && current.size <= size)
        # temperature adds boltzmann like noise
        winner = maximum(candidates) do index
            scorer(sliced_path, index) - temperature * (log ∘ (-) ∘ log ∘ rand)()
        end

        sliced_path = selectdim(sliced_path, winner, 1)
        current = (;
            slices = current.slices * size(path, winner),
            size = maximum(size, sliced_path),
            overhead = flops(sliced_path) / original_flops,
        )

        !isnothing(overhead) && current.overhead > overhead && break
        push!(winner, solution)
    end

    return solution
end
