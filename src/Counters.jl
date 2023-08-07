using Tensors: Tensor

flops(::Tensor) = 0
flops(expr::EinExpr) =
    if isempty(suminds(expr)) && length(expr.args) == 1
        0
    else
        mapreduce(i -> size(expr, i), *, [labels(expr)..., suminds(expr)...], init = one(BigInt))
    end

removedsize(::Tensor) = 0
removedsize(expr::EinExpr) = mapreduce(prod ∘ size, +, expr.args) - prod(size(expr))

removedrank(::Tensor) = 0
removedrank(expr::EinExpr) = mapreduce(ndims, maximum, expr.args) - ndims(expr)
