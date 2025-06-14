using CliqueTrees: MF, MMD, ND, SafeRules, KaHyParND, METISND

"""
    HyPar(dis, algs;
        level = 6,
        width = 120,
        imbalances = 130:130,
    )

Nested-dissection based optimizer. Recursively partitions a tensor network, then calls a
greedy algorithm on the leaves. The optimizer is run a number of times: once for each greedy
algorithm in `algs` and each imbalance value in `imbalances`. The recursion depth is controlled by
the parameters `level` and `width`.

The line graph is partitioned using the algorithm `dis`. EinExprs currently supports two partitioning
algorithms, both of which require importing an external library.

| type                                                                                             | package                                              |
|:-------------------------------------------------------------------------------------------------|:-----------------------------------------------------|
| [`METISND`](https://algebraicjulia.github.io/CliqueTrees.jl/stable/api/#CliqueTrees.METISND)     | [Metis.jl](https://github.com/JuliaSparse/Metis.jl)  |
| [`KaHyParND`](https://algebraicjulia.github.io/CliqueTrees.jl/stable/api/#CliqueTrees.KaHyParND) | [KayHyPar.jl](https://github.com/kahypar/KaHyPar.jl) |

The optimizer is implemented using the tree decomposition library
[CliqueTrees.jl](https://github.com/AlgebraicJulia/CliqueTrees.jl).

# Arguments

  - `dis`: [graph partitioning algorithm](https://algebraicjulia.github.io/CliqueTrees.jl/stable/api/#CliqueTrees.DissectionAlgorithm)
  - `algs`: tuple of [elimination algorithms](https://algebraicjulia.github.io/CliqueTrees.jl/stable/api/#Elimination-Algorithms).
  - `level`: maximum level
  - `width`: minimum width
  - `imbalances`: imbalance parameters 

"""
struct HyPar{D, A} <: Optimizer
    dis::D
    algs::A
    level::Int
    width::Int
    imbalances::StepRange{Int, Int}
end

function HyPar(dis::D=KaHyParND(), algs::A = (MF(), MMD());
        level::Integer = 6,
        width::Integer = 120,
        imbalances::AbstractRange=130:130,
    ) where{D, A}

    return HyPar{D, A}(dis, algs, level, width, imbalances)
end

# scoring function used during
# hyperparameter search
function score(path::SizedEinExpr)
    return log2(mapreduce(flops, +, Branches(path)))
end

function EinExprs.einexpr(config::HyPar, path)
    dis = config.dis
    algs = config.algs
    level = config.level
    width = config.width
    imbalances = config.imbalances

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

function Base.show(io::IO, ::MIME"text/plain", config::HyPar{D, A}) where {D, A}
    println(io, "HyPar{$D, $A}:")
    show(IOContext(io, :indent => 4), "text/plain", config.dis)

    for alg in config.algs
        show(IOContext(io, :indent => 4), "text/plain", alg)
    end

    println(io, "    level: $(config.level)")
    println(io, "    width: $(config.width)")
    println(io, "    imbalances: $(config.imbalances)")
    return
end
