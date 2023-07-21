using Tensors: Tensor

flops(::Tensor) = 0
flops(expr::EinExpr) = mapreduce(i -> size(expr, i), *, [labels(expr)..., suminds(expr)...])

removedsize(::Tensor) = 0
removedsize(expr::EinExpr) = mapreduce(prod ∘ size, +, expr.args) - prod(size(expr))

removedrank(::Tensor) = 0
removedrank(expr::EinExpr) = mapreduce(ndims, maximum, expr.args) - ndims(expr)
