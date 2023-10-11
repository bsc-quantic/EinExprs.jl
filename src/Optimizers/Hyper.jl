using SparseArrays
using KaHyPar

@kwdef struct HyperGraphPartitioning <: Optimizer
    metric::Function = removedsize
    outer::Bool = false
end


function get_hypergraph(path)
    all_indices = inds(path) # Assume unique indices across all tensors

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

function recursive_partition(expr::EinExpr; parts=2, parts_decay=0.5, random_strength=0.01, cutoff=2, max_iterations=10)
    # Convert the EinExpr to a hypergraph
    hypergraph = get_hypergraph(expr)
    println("Hypergraph size: ", hypergraph.n_vertices)

    # Base case: if the hypergraph is small enough, we stop the recursion
    if hypergraph.n_vertices <= cutoff
        println("Base case")
        return expr
    end

    if max_iterations == 0
        println("Max iterations reached")
        return expr
    end

    # Partition the hypergraph
    partitioning_result = partition_hypergraph(hypergraph, parts=parts, parts_decay=parts_decay, random_strength=random_strength)

    # Get unique partitions
    unique_partitions = unique(partitioning_result)

    # Contract nodes based on the partitioning
    new_exprs = [partition_to_einexpr(part_id, partitioning_result, expr) for part_id in unique_partitions]
    combined_expr = sum(new_exprs)

    # Recursively call the function on the combined EinExpr
    return recursive_partition(combined_expr, parts=parts, parts_decay=parts_decay, random_strength=random_strength, cutoff=cutoff, max_iterations=max_iterations-1)
end

function partition_to_einexpr(partition_id::Int, partition_result::Vector{Int}, original_path::EinExpr)
    # Identify the tensor indices that belong to the specific partition_id
    tensor_indices_in_partition = [idx for (idx, part) in enumerate(partition_result) if part == partition_id]

    # Extract the corresponding tensors from the original EinExpr
    tensors_in_partition = [original_path.args[idx] for idx in tensor_indices_in_partition]

    # Sum (contract) these tensors to create a new tensor
    new_tensor = sum(tensors_in_partition)

    # @show tensors_in_partition
    # @show tensor_indices_in_partition
    # @show new_tensor

    # Create a new EinExpr with the new tensor's indices in the head, the original tensors in the args, and the size dictionary
    new_einexpr = EinExpr(new_tensor.head, tensors_in_partition)

    return new_einexpr
end
