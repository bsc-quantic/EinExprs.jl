using AbstractTrees
using SparseArrays
using KaHyPar

@kwdef struct HyPar <: Optimizer
    parts = 2
    parts_decay = 0.5
    random_strength = 0.01
    cutoff = 2
    max_iterations = 100
end

function EinExprs.einexpr(config::HyPar, path)
    inds = mapreduce(head, ∪, path.args)
    indexmap = Dict(Iterators.map(splat(Pair) ∘ reverse, enumerate(inds)))

    I = Iterators.flatmap(tensor -> fill(1, ndims(tensor)), path.args) |> collect
    J = Iterators.flatmap(tensor -> Iterators.map(Base.Fix1(getindex, indexmap), head(tensor)), path.args) |> collect
    V = fill(1, length(I))
    incidence_matrix = sparse(I, J, V)

    # NOTE indices in `inds` should be in the same order as unique indices appear by iterating on `path.args` because `∪` retains order
    edge_weights = map(Base.Fix1(size, path), inds)
    vertex_weights = ones(Int, length(path.args))

    hypergraph = KaHyPar.HyperGraph(incidence_matrix, vertex_weights, edge_weights)

    for iteration in 1:config.max_iterations
        # base case: no more children
        path.args == EinExpr[] && break

        # Convert the EinExpr to a hypergraph
        hypergraph = get_hypergraph(path)

        # Check if hypergraph size is below the cutoff
        if hypergraph.n_vertices <= cutoff
            println("Hypergraph size below cutoff.")
            break
        end

        # Partition the hypergraph
        # The result is a vector of partition ids for each vertex
        partitioning_result = partition_hypergraph(
            hypergraph,
            parts = parts,
            parts_decay = parts_decay,
            random_strength = random_strength,
        )

        # Get unique partitions
        unique_partitions = unique(partitioning_result)

        # Contract nodes based on the partitioning
        new_exprs = [partition_to_einexpr(part_id, partitioning_result, path) for part_id in unique_partitions]
        combined_expr = sum(new_exprs)

        # Update the expression for the next iteration
        path = combined_expr
    end

    return path
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
