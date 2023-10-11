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


function partition_hypergraph(hypergraph::KaHyPar.HyperGraph; parts=2, parts_decay=0.5, random_strength=0.01)
    # Create a context for KaHyPar
    context = KaHyPar.kahypar_context_new()

    # Load default configuration
    config_file_path = KaHyPar.default_configuration
    KaHyPar.kahypar_configure_context_from_file(context, config_file_path)

    # Calculate the relative subgraph size
    subsize = hypergraph.n_vertices
    N = hypergraph.n_vertices  # Assuming N is the total number of vertices in the entire graph
    s = subsize / N

    # Determine the number of partitions based on the relative subgraph size
    kparts = max(Int(s^parts_decay * parts), 2)

    # Perform the partitioning
    partitioning_result = KaHyPar.partition(hypergraph, kparts; configuration=config_file_path)

    # Clean up the context
    KaHyPar.kahypar_context_free(context)

    return partitioning_result
end



function recursive_partition(hypergraph::KaHyPar.HyperGraph; parts=2, parts_decay=0.5, random_strength=0.01, cutoff=10)
    # Base case: if the hypergraph is small enough, we stop the recursion
    if hypergraph.n_vertices <= cutoff
        return hypergraph
    end

    # Partition the hypergraph
    partitioning_result = partition_hypergraph(hypergraph, parts=parts, parts_decay=parts_decay, random_strength=random_strength)

    # Contract nodes based on the partitioning
    # For simplicity, we'll just merge nodes in the same partition into a single node for now
    contracted_hypergraph = contract_nodes(hypergraph, partitioning_result)

    # Recursively call the function on the contracted hypergraph
    return recursive_partition(contracted_hypergraph, parts=parts, parts_decay=parts_decay, random_strength=random_strength, cutoff=cutoff)
end

function contract_nodes(hypergraph::KaHyPar.HyperGraph, partitioning_result)
    # This is a placeholder and you'll need to implement the actual contraction logic

    return hypergraph  # Placeholder
end
