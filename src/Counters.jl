flops(expr::EinExpr) =
    if length(expr.args) == 0
        0
    else
        mapreduce(i -> size(expr, i), *, [head(expr)..., suminds(expr)...], init = one(BigInt))
    end

removedsize(expr::EinExpr) = mapreduce(prod ∘ size, +, expr.args) - prod(size(expr))

removedrank(expr::EinExpr) = mapreduce(ndims, maximum, expr.args) - ndims(expr)
