using Tensors: Tensor
using Memoize

flops(::Tensor) = 0
@memoize function flops(expr::EinExpr)
    flops_sub = sum(flops.(expr.args))

    floppi(inds) = mapreduce(i -> size(expr, i), *, inds, init = one(BigInt))
    flops_cur = floppi(suminds(expr)) * (isempty(suminds(expr)) && length(expr.args) == 1 ? 0 : floppi(labels(expr)))

    return flops_sub + flops_cur
end

removedsize(::Tensor) = 0
removedsize(expr::EinExpr) = mapreduce(prod ∘ size, +, expr.args) - prod(size(expr))

removedrank(::Tensor) = 0
removedrank(expr::EinExpr) = mapreduce(ndims, maximum, expr.args) - ndims(expr)
