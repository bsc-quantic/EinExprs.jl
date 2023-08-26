flops(expr::EinExpr) =
    if isempty(suminds(expr)) && length(expr.args) == 1
        0
    else
        mapreduce(i -> size(expr, i), *, [head(expr)..., suminds(expr)...], init = one(BigInt))
    end

removedsize(expr::EinExpr) = mapreduce(prod ∘ size, +, expr.args) - prod(size(expr))

removedrank(expr::EinExpr) = mapreduce(ndims, maximum, expr.args) - ndims(expr)
