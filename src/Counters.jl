using Tensors: Tensor
using Memoize

Base.ndims(expr::EinExpr) = length(labels(expr))

flops(::Tensor) = 0
@memoize function flops(expr::EinExpr)
    flops_sub = sum(flops.(expr.args))

    floppi(inds) = mapreduce(i -> size(expr, i), *, inds, init=one(BigInt))
    flops_cur = floppi(suminds(expr)) * (isempty(suminds(expr)) && length(expr.args) == 1 ? 0 : floppi(labels(expr)))

    return flops_sub + flops_cur
end

removedsize(::Tensor) = 0
removedsize(expr::EinExpr) = mapreduce(prod ∘ size, +, expr.args) - prod(size(expr))
