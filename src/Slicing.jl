Base.selectdim(path::EinExpr, index::Symbol, i) = EinExpr(map(path.args) do sub
        index ∈ __labels_children(sub) ? selectdim(sub, index, i) : sub
    end, filter(!=(index), path.head))

__labels_children(x) = labels(x)
__labels_children(path::EinExpr) = labels(path, all=true)

Base.view(path::EinExpr, cuttings::Pair{Symbol,<:Integer}...) =
    reduce(cuttings, init=path) do acc, proj
        d, i = proj
        selectdim(acc, d, i)
    end

function slices(
    target::Function,
    path::EinExpr;
    size=nothing,
    overhead=nothing,
    slices=nothing,
    temperature=0.01,
    skip=Set{Symbol}()
)
    candidates = setdiff(labels(path, all=true), skip)
    solution = Set{Symbol}()

    current = (; slices=1, size=..., overhead=1.0)

    checkpredicates() = !isnothing(size) && ... || !isnothing(slices) && ... || !isnothing(overhead) && ...

    while checkpredicates()
        winner = maximum(candidates) do index
            # score + boltzmann sampling
            target(...) - temperature * (log ∘ (-) ∘ log ∘ rand)
        end

        push!(winner, solution)
        current = (;
            slices=current.slices * size(path, winner),
            size=...,
            overhead=...
        )
    end

    return solution
end
