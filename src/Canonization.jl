"""
    Canonization

Abstract type of canonization methods: `EinExpr` transformations that ease or prepare the path to be then optimized by a `Optimizer` or that simplify the path for further processing.
"""
abstract type Canonization end

canonize(method::Canonization, path::EinExpr; kwargs...) = canonize!(method, deepcopy(path); kwargs...)

function canonize! end

"""
    SumShadowInds

Sums over shadow indices; i.e. indices whose size is 1.
"""
struct SumShadowInds <: Canonization end

function canonize!(::SumShadowInds, path::SizedEinExpr)
    shadowinds = filter(inds(path)) do i
        size(path, i) == 1
    end

    for i in shadowinds
        sum!(path, i)
    end

    return path
end

"""
    SumOpenInds

Sums over open indices that do not appear in the output; i.e. indices that appear only once but should be summed.
"""
struct SumOpenInds <: Canonization end

function canonize!(::SumOpenInds, path::SizedEinExpr)
    targets = setdiff(openinds(path), head(path))

    for i in targets
        sum!(path, i)
    end

    return path
end

"""
    Binarize

Binarize an n-ary contraction tree.
"""
struct Binarize <: Canonization end

function canonize!(::Binarize, path::EinExpr)
    if length(args(path)) > 2
        copy!(args(path), args(einexpr(Naive(), path)))
    end

    for arg in args(path)
        canonize!(Binarize(), arg)
    end

    return path
end

function canonize!(::Binarize, path::SizedEinExpr)
    canonize!(Binarize(), path.path)
    return path
end
