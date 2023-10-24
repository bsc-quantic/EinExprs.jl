module EinExprsTensorOperationsExt

using EinExprs
using TensorOperations
using TensorOperations: optimaltree

@kwdef struct Netcon <: Optimizer
    outer::Bool = false
end

function EinExprs.einexpr(::Netcon, path::EinExpr; verbose = false)
    # convert to tensor operations format
    network = [head(expr) for expr in leaves(path)]
    optdata = Dict(i => size(path, i) for i in inds(path))

    tree, cost = optimaltree(network, optdata)
    verbose && @info "Optimal contraction cost: $cost"
    # convert back to einexprs format
    tree_to_path(i::Int) = leaves(path, i)
    tree_to_path(tree::Vector) = sum(map(tree_to_path, tree))

    return EinExpr(head(path), map(tree_to_path, tree))
end

function TensorOperations.ncon(tensors, network::EinExpr, conjlist = fill(false, length(tensors)))
    length(tensors) == length(leaves(network)) == length(conjlist) ||
        throw(ArgumentError("number of tensors and of index lists should be the same"))

    # Special case for single tensor
    if length(tensors) == 1
        conjflag = conjlist[1] ? :C : :N
        if isempty(network.args) # copy with optional conj
            return tensorcopy(head(network), tensors[1], head(network), conjflag)
        elseif length(network.args) == 1 # trace or copy/permute
            if isempty(suminds(network)) # copy/permute
                return tensorcopy(head(network), tensors[1], head(network.args[1]), conjflag)
            else # trace
                return tensortrace(head(network), tensors[1], head(network.args[1]), conjflag)
            end
        else
            throw(ArgumentError("Invalid network: $network"))
        end
    end

    @assert length(args(network)) == 2 "Only binary trees are supported"
    A, CA = contracttree(tensors, network, conjlist, args(network)[1])
    B, CB = contracttree(tensors, network, conjlist, args(network)[2])

    return tensorcontract(head(network), A, head(args(network)[1]), CA, B, head(args(network)[2]), CB)
end

function contracttree(tensors, network, conjlist, tree)
    @nospecialize
    if length(args(tree)) == 0 # leaf
        # this is a bit of a hack to find the correct input tensor, but I don't know how to do it better without changing the API
        tensor_id = findfirst(==(tree), leaves(network))
        return tensors[tensor_id], conjlist[tensor_id] ? :C : :N
    elseif length(args(tree)) == 1 # trace
        A, CA = contracttree(tensors, network, conjlist, args(tree)[1])
        return tensortrace!(head(tree), A, head(args(tree)[1]), CA), :N
    elseif length(args(tree)) == 2 # binary contraction
        A, CA = contracttree(tensors, network, conjlist, args(tree)[1])
        B, CB = contracttree(tensors, network, conjlist, args(tree)[2])

        return tensorcontract(head(tree), A, head(args(tree)[1]), CA, B, head(args(tree)[2]), CB), :N
    else
        throw(ArgumentError("Only binary trees are supported"))
    end
end

end
