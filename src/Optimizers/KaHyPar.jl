using AbstractTrees
using SparseArrays
using KaHyPar

@kwdef struct HyPar <: Optimizer
    parts = 2
    imbalance = 0.03
    cutoff = 2
    configuration::Union{Nothing,Symbol,String} = nothing
end

function EinExprs.einexpr(config::HyPar, path)
    inds = mapreduce(head, ∪, path.args)
    indexmap = Dict(Iterators.map(splat(Pair) ∘ reverse, enumerate(inds)))

    I = Iterators.flatmap(((i, tensor),) -> fill(i, ndims(tensor)), enumerate(path.args)) |> collect
    J = Iterators.flatmap(tensor -> Iterators.map(Base.Fix1(getindex, indexmap), head(tensor)), path.args) |> collect
    V = fill(1, length(I))
    incidence_matrix = sparse(I, J, V)

    # NOTE indices in `inds` should be in the same order as unique indices appear by iterating on `path.args` because `∪` retains order
    edge_weights = map(Base.Fix1(size, path), inds)
    vertex_weights = ones(Int, length(path.args))

    hypergraph = KaHyPar.HyperGraph(incidence_matrix, vertex_weights, edge_weights)

    # stop on cutoff
    hypergraph.n_vertices <= config.cutoff && return path

    partitions =
        KaHyPar.partition(hypergraph, config.parts; imbalance = config.imbalance, configuration = config.configuration)

    args = map(unique(partitions)) do partition
        expr = sum(path.args[partitions.==partition], skip = path.head)
        einexpr(config, expr)
    end

    return EinExpr(path.head, args)
end
