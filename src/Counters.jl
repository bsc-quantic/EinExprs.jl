using Tensors: Tensor

flops(::Tensor) = 0
function flops(expr::EinExpr)
    flops_sub = sum(flops.(expr.args))

    floppi(inds) = mapreduce(i -> size(expr, i), *, inds, init=one(BigInt))
    flops_cur = floppi(suminds(expr)) * (isempty(suminds(expr)) && length(expr.args) == 1 ? 0 : floppi(labels(expr)))

    return flops_sub + flops_cur
end
