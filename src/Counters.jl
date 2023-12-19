"""
    flops(path::EinExpr)

Count the number of mathematical operations will be performed by the contraction of the root of the `path` tree.
"""
flops(sexpr::SizedEinExpr) =
    if nargs(sexpr) == 0 || nargs(sexpr) == 1 && isempty(suminds(sexpr))
        0
    else
        mapreduce(
            Base.Fix1(getindex, sexpr.size),
            *,
            Iterators.flatten((head(sexpr), suminds(sexpr))),
            init = one(BigInt),
        )
    end

flops(expr::EinExpr, size) = flops(SizedEinExpr(expr, size))

"""
    removedsize(path::EinExpr)

Count the amount of memory that will be freed after performing the contraction of the root of the `path` tree.
"""
function removedsize(sexpr::SizedEinExpr)
    mapreduce(prod ∘ Base.Fix2(size, sexpr.size), +, sexpr.args) - prod(size(sexpr, sexpr.size))
end

removedsize(expr::EinExpr, size) = removedsize(SizedEinExpr(expr, size))

"""
    removedrank(path::EinExpr)

Count the rank reduction after performing the contraction of the root of the `path` tree.
"""
removedrank(expr::EinExpr) = mapreduce(ndims, max, expr.args) - ndims(expr)
removedrank(expr::EinExpr, _) = removedrank(expr)
removedrank(sexpr::SizedEinExpr, _) = removedrank(sexpr.path)

for f in [:flops, :removedsize]
    @eval $f(sizedict::Dict{Symbol}) = Base.Fix2($f, sizedict)
end
removedrank(::Dict) = removedrank
