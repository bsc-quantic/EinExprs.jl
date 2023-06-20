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