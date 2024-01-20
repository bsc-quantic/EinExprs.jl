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

function flops(_out, _suminds, sizelist)
    mapreduce(*, enumerate(sizelist)) do (i, size)
        onehot_in(i, _out) || onehot_in(i, _suminds) ? size : 1
    end
end

function fastflops(sexpr::SizedEinExpr)
    if nargs(sexpr) == 0 || nargs(sexpr) == 1 && isempty(suminds(sexpr))
        return 0
    end

    mapreduce(
        log ∘ Base.Fix1(getindex, sexpr.size),
        +,
        Iterators.flatten((head(sexpr), suminds(sexpr))),
        init = zero(Float64),
    ) |>
    exp |>
    round |>
    BigInt
end

fastflops(expr::EinExpr, size) = fastflops(SizedEinExpr(expr, size))

"""
    removedsize(path::EinExpr)

Count the amount of memory that will be freed after performing the contraction of the root of the `path` tree.
"""
removedsize(sexpr::SizedEinExpr) = -length(sexpr) + mapreduce(+, sexpr.args) do arg
    length(SizedEinExpr(arg, sexpr.size))
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
    @eval $f(sizedict::Dict) = Base.Fix2($f, sizedict)
end
removedrank(::Dict) = removedrank
