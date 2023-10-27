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

    num_columns = maximum(values(indexmap))
    num_rows = length(path.args)
    incidence_matrix = spzeros(Int, num_rows, num_columns)

    # Iterate through each tensor and its associated indices, and update the incidence matrix.
    for (i, tensor) in enumerate(path.args)
        tensor_indices = [i]  # Current tensor is represented as a row in the matrix.
        edge_indices = [indexmap[idx] for idx in head(tensor)]  # Map indices via 'indexmap'.

        # Create a subview for the current tensor and associated hyperedges.
        incidence_subview = view(incidence_matrix, tensor_indices, edge_indices)
        incidence_subview .= 1  # Update the subview directly. This step modifies the original sparse matrix.
    end

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
