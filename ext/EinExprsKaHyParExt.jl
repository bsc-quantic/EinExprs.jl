module EinExprsKaHyParExt

using CliqueTrees
using EinExprs

function score(path::SizedEinExpr)
    return log2(mapreduce(flops, +, Branches(path)))
end

function EinExprs.einexpr(config::EinExprs.HyPar, path)
    algs = config.algs
    level = config.level
    width = config.width
    imbalances = config.imbalances

    dis = KaHyParND()
    minpath = nothing; minscore = typemax(Float64)

    for alg in algs, imbalance in imbalances

        curconfig = LineGraph(SafeRules(ND(alg, dis;
            level,
            width,
            imbalance,
        )))

        curpath = einexpr(curconfig, path)
        curscore = score(curpath)

        if curscore < minscore
            minpath, minscore = curpath, curscore
        end
    end

    return minpath
end

end
