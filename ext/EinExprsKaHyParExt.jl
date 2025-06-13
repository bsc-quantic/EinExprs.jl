module EinExprsKaHyParExt

using CliqueTrees
using EinExprs

function score(path::SizedEinExpr)
    return log2(mapreduce(flops, +, Branches(path)))
end

function EinExprs.einexpr(config::EinExprs.HyPar, path)
    algs = (
        MF(),
        MMD(),
    )

    diss = (
        KaHyParND(),
    )

    imbalances = config.imbalances
    minpath = nothing; minscore = typemax(Float64)

    for alg in algs, dis in diss, imbalance in imbalances
        curconfig = LineGraph(SafeRules(ND(alg, dis; imbalance)))
        curpath = einexpr(curconfig, path)
        curscore = score(curpath)

        if curscore < minscore
            minpath, minscore = curpath, curscore
        end
    end

    return minpath
end

end
