"""
    flops(path::EinExpr)

Count the number of mathematical operations will be performed by the contraction of the root of the `path` tree.
"""
flops(expr::EinExpr, sizedict) =
    if length(expr.args) == 0 || length(expr.args) == 1 && isempty(suminds(expr))
        0
    else
        mapreduce(i -> sizedict[i], *, Iterators.flatten((head(expr), suminds(expr))), init = one(BigInt))
    end

"""
    removedsize(path::EinExpr)

Count the amount of memory that will be freed after performing the contraction of the root of the `path` tree.
"""
function removedsize(expr::EinExpr, sizedict)
    mapreduce(prod ∘ Base.Fix2(size, sizedict), +, expr.args) - prod(size(expr, sizedict))
end

"""
    removedrank(path::EinExpr)

Count the rank reduction after performing the contraction of the root of the `path` tree.
"""
removedrank(expr::EinExpr, _) = mapreduce(ndims, max, expr.args) - ndims(expr)

for f in [:flops, :removedsize]
    @eval $f(sizedict::Dict{Symbol}) = Base.Fix2($f, sizedict)
end
removedrank(::Dict) = Base.Fix2(removedrank, nothing)
