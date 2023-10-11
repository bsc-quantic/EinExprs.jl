@kwdef struct HyperOptimized <: Optimizer
    metric::Function = removedsize
    outer::Bool = false
end

using SparseArrays
using KaHyPar

function get_hypergraph(path)
    all_indices = path.head # Assume unique indices across all tensors

    # Create incidence matrix
    incidence_matrix = spzeros(Int64, length(path.args), length(all_indices))

    for (i, tensor) in enumerate(path.args)
        for idx in tensor.head
            j = findfirst(==(idx), all_indices)
            incidence_matrix[i, j] = 1
        end
    end

    # Vertex weights (assuming equal weight for all tensors)
    vertex_weights = ones(Int64, length(path.args))

    # Hyperedge weights set to the size of the index
    edge_weights = [size(path, idx) for idx in all_indices]

    # Create hypergraph
    hypergraph = KaHyPar.HyperGraph(incidence_matrix, vertex_weights, edge_weights)

    return hypergraph
end

