using AbstractTrees
using SparseArrays
using KaHyPar

@kwdef struct HyPar <: Optimizer
    parts::Int = 2
    imbalance::Float32 = 0.03
    stop::Function = <=(2) ∘ length ∘ Base.Fix1(getfield, :args)
    configuration::Union{Nothing,Symbol,String} = nothing
    edge_scaler::Function = (ind_size) -> 1000 * (Int ∘ round ∘ log2)(ind_size)
    vertex_scaler::Function = (prod_size) -> 1000 * (Int ∘ round ∘ log2)(prod_size)
end

function EinExprs.einexpr(config::HyPar, path)
    config.stop(path) && return path

    inds = mapreduce(head, ∪, path.args)
    indexmap = Dict(Iterators.map(splat(Pair) ∘ reverse, enumerate(inds)))

    I = Iterators.flatmap(((i, tensor),) -> fill(i, ndims(tensor)), enumerate(path.args)) |> collect
    J = Iterators.flatmap(tensor -> Iterators.map(Base.Fix1(getindex, indexmap), head(tensor)), path.args) |> collect
    V = fill(1, length(I))
    incidence_matrix = sparse(I, J, V)

    # NOTE indices in `inds` should be in the same order as unique indices appear by iterating on `path.args` because `∪` retains order
        edge_weights = map(ind -> (config.edge_scaler ∘ Base.Fix1(size, path))(ind), inds)
        vertex_weights = map(config.vertex_scaler ∘ length, path.args)

    hypergraph = KaHyPar.HyperGraph(incidence_matrix, vertex_weights, edge_weights)

    partitions =
        KaHyPar.partition(hypergraph, config.parts; imbalance = config.imbalance, configuration = config.configuration)

    args = map(unique(partitions)) do partition
        selection = partitions .== partition
        count(selection) == 1 && return only(path.args[selection])

        expr = sum(path.args[selection], skip = path.head)
        einexpr(config, expr)
    end

    return EinExpr(path.head, args)
end
