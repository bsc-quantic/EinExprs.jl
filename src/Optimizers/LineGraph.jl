using AbstractTrees
using Base: oneto
using CliqueTrees: EliminationAlgorithm, CompositeRotations, MF, cliquetree, residual, separator
using SparseArrays

struct LineGraph{A <: EliminationAlgorithm} <: Optimizer
    alg::A
end

function LineGraph()
    # the default algorithm is the
    # weighted min-fill heuristic
    return LineGraph(MF())
end

function einexpr(config::LineGraph, path::EinExpr{L}, sizedict::Dict{L}) where {L}
    tensors = leaves(path)

    # construct hypergraph
    il = L[]; li = Dict{L, Int}()
    weights = Float64[]; nzval = Int[]; rowval = Int[]
    colptr = Int[]; push!(colptr, 1)

    for tensor in tensors
        for l in head(tensor)
            if !haskey(li, l)
                push!(weights, log2(sizedict[l]))
                push!(il, l); li[l] = length(il)
            end

            push!(nzval, 1)
            push!(rowval, li[l])
        end

        push!(colptr, length(rowval) + 1)
    end

    for l in head(path)
        push!(nzval, 1)
        push!(rowval, li[l])
    end

    push!(colptr, length(rowval) + 1)
    m = length(il); n = length(tensors)
    it = SparseMatrixCSC{Int, Int}(m, n + 1, colptr, rowval, nzval)
    ti = copy(transpose(it))

    # construct tree decomposition
    clique = view(rowvals(it), nzrange(it, n + 1))
    alg = CompositeRotations(clique, config.alg)
    perm, tree = cliquetree(weights, ti' * ti; alg)
    
    # permute hypergraph
    permute!(il, perm)
    permute!(ti, oneto(n + 1), perm)

    # compute subtree roots
    roots = Vector{Int}(undef, m)

    for (b, bag) in enumerate(tree), i in residual(bag)
        roots[i] = b
    end

    # construct expression
    tags = zeros(Bool, n + 1); tags[end] = true
    stack = EinExpr{L}[]

    for (b, bag) in enumerate(tree)
        tensor = EinExpr(L[])

        for i in separator(bag)
            push!(head(tensor), il[i])
        end
        
        for i in residual(bag), t in view(rowvals(ti), nzrange(ti, i))
            if !tags[t]
                tags[t] = true
                push!(args(tensor), tensors[t])
            end
        end

        for _ in childindices(tree, b)
            push!(args(tensor), pop!(stack))
        end

        push!(stack, tensor)
    end

    result = first(stack)

    for tensor in stack[2:end]
        append!(args(result), args(tensor))
    end

    append!(head(result), head(path))
    return result
end

function einexpr(config::LineGraph, path::SizedEinExpr)
    return SizedEinExpr(einexpr(config, path.path, path.size), path.size)
end
