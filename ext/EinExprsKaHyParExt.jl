module EinExprsKaHyParExt

using EinExprs
using AbstractTrees
using SparseArrays
using KaHyPar
using Suppressor
using Compat

function EinExprs.einexpr(config::EinExprs.HyPar, path)
    config.stop(path) && return path

    inds = mapreduce(head, ∪, path.args)
    indexmap = Dict(Iterators.map(@compat(splat(Pair)) ∘ reverse, enumerate(inds)))

    I = EinExprs.flatmap(((i, tensor),) -> fill(i, ndims(tensor)), enumerate(path.args)) |> collect
    J = EinExprs.flatmap(tensor -> Iterators.map(Base.Fix1(getindex, indexmap), head(tensor)), path.args) |> collect
    V = fill(1, length(I))
    incidence_matrix = sparse(I, J, V)

    # NOTE indices in `inds` should be in the same order as unique indices appear by iterating on `path.args` because `∪` retains order
    edge_weights = map(config.edge_scaler ∘ Base.Fix1(size, path), inds)
    vertex_weights = map(config.vertex_scaler ∘ length, args(path))

    hypergraph = KaHyPar.HyperGraph(incidence_matrix, vertex_weights, edge_weights)
    KaHyPar.kahypar_set_seed(hypergraph.context, config.seed)

    partitions = @suppress KaHyPar.partition(
        hypergraph,
        config.parts;
        imbalance = config.imbalance,
        configuration = config.configuration,
    )

    _args = map(unique(partitions)) do partition
        selection = partitions .== partition
        count(selection) == 1 && return only(args(path)[selection])

        expr = sum(args(path)[selection], skip = path.head)
        einexpr(config, expr)
    end

    return sum(_args, skip = path.head)
end

end
