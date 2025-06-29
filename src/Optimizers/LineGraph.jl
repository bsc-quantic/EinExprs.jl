using AbstractTrees
using Base: oneto
using CliqueTrees: EliminationAlgorithm, MF, cliquetree, cliquetree!, residual, separator
using SparseArrays

"""
    LineGraph(alg::EliminationAlgorithm)

Tree decomposition based solver. Constructs the line graph of a tensor networks and finds a tree
decomposition using [CliqueTrees.jl](https://github.com/AlgebraicJulia/CliqueTrees.jl).

# Arguments

  - `alg` is an elimination algorithm. See the list [here](https://algebraicjulia.github.io/CliqueTrees.jl/stable/api/#Elimination-Algorithms).
"""
struct LineGraph{A<:EliminationAlgorithm} <: Optimizer
    alg::A
end

function LineGraph()
    # the default algorithm is the
    # weighted min-fill heuristic
    return LineGraph(MF())
end

function einexpr(config::LineGraph, path::EinExpr{L}, sizedict::Dict{L}) where {L}
    tensors = leaves(path)

    # construct incidence matrix `ti`
    #           indices
    #         [         ]
    # tensors [    ti   ]
    #         [         ]
    # we only care about the sparsity pattern
    il = L[];
    li = Dict{L,Int}() # il ∘ li = id
    weights = Float64[];
    nzval = Int[];
    rowval = Int[]
    colptr = Int[];
    push!(colptr, 1)

    for tensor in tensors
        for l in head(tensor)
            if !haskey(li, l)
                push!(weights, log2(sizedict[l]))
                push!(il, l);
                li[l] = length(il)
            end

            push!(nzval, 1)
            push!(rowval, li[l])
        end

        push!(colptr, length(rowval) + 1)
    end

    # add a "virtual" tensor with indices `head(path)`
    for l in head(path)
        push!(nzval, 1)
        push!(rowval, li[l])
    end

    push!(colptr, length(rowval) + 1)
    m = length(il);
    n = length(tensors)
    it = SparseMatrixCSC{Int,Int}(m, n + 1, colptr, rowval, nzval)
    ti = copy(transpose(it))

    # construct line graph `ii`
    #           indices
    #         [         ]
    # indices [    ii   ]
    #         [         ]
    # we only care about the sparsity pattern
    ii = ti' * ti

    # compute a tree (forest) decomposition of `ii`
    perm, tree = cliquetree(weights, ii; alg = config.alg)

    # find the bag containing `clique`, call it `root`
    clique = zeros(Bool, m);
    root = 0

    for i in view(rowvals(it), nzrange(it, n + 1))
        clique[i] = true
    end

    for (b, bag) in enumerate(tree)
        !iszero(root) && break

        for i in residual(bag)
            !iszero(root) && break

            if clique[perm[i]]
                root = b
            end
        end
    end

    # make `root` a root node of the tree decomposition
    permute!(perm, cliquetree!(tree, root))

    # permute incidence matrix `ti`
    permute!(il, perm)
    permute!(ti, oneto(n + 1), perm)

    # the vector `roots` maps each vertex to the root node
    # of its subtree
    roots = Vector{Int}(undef, m)

    for (b, bag) in enumerate(tree), i in residual(bag)
        roots[i] = b
    end

    # dynamic programming
    tags = zeros(Bool, n + 1);
    stack = EinExpr{L}[]

    for (b, bag) in enumerate(tree)
        sep = separator(bag)
        res = residual(bag)
        tensor = EinExpr(il[sep])

        for i in res, t in view(rowvals(ti), nzrange(ti, i))
            if !tags[t]
                tags[t] = true

                if t > n
                    append!(head(tensor), head(path))
                else
                    push!(args(tensor), tensors[t])
                end
            end
        end

        for _ in childindices(tree, b)
            push!(args(tensor), pop!(stack))
        end

        push!(stack, tensor)
    end

    # we now have an expression for each root
    # of the tree decomposition
    if isone(length(stack))
        result = only(stack)
    else
        result = EinExpr(copy(head(path)), stack)
    end

    return canonize!(Binarize(), result)
end

function einexpr(config::LineGraph, path::SizedEinExpr)
    return SizedEinExpr(einexpr(config, path.path, path.size), path.size)
end
