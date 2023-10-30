"""
    flops(path::EinExpr)

Count the number of mathematical operations will be performed by the contraction of the root of the `path` tree.
"""
flops(expr::EinExpr) =
    if length(expr.args) == 0 || length(expr.args) == 1 && isempty(suminds(expr))
        0
    else
        mapreduce(Base.Fix1(size, expr), *, Iterators.flatten((head(expr), suminds(expr))), init = one(BigInt))
    end

"""
    removedsize(path::EinExpr)

Count the amount of memory that will be freed after performing the contraction of the root of the `path` tree.
"""
removedsize(expr::EinExpr) = mapreduce(prod ∘ size, +, expr.args) - prod(size(expr))

"""
    removedrank(path::EinExpr)

Count the rank reduction after performing the contraction of the root of the `path` tree.
"""
removedrank(expr::EinExpr) = mapreduce(ndims, max, expr.args) - ndims(expr)
