using Base: @kwdef
using CliqueTrees: MF, MMD

"""
    HyPar(algs;
        level = 6,
        width = 120,
        imbalances = 130:130,
    )

Hypergraph partitioning based optimizer. Recursively bipartitions a tensor network, then calls a
greedy algorithm on the leaves. The optimizer is run a number of times: once for each greedy
algorithm in `algs` and each imbalance value in `imbalances`. The recursion depth is controlled by
the parameters `level` and `width`.

The optimizer is implemented using the hypergraph partitioning
library [KaHyPar.jl](https://github.com/kahypar/KaHyPar.jl) and the tree decomposition library
[CliqueTrees.jl](https://github.com/AlgebraicJulia/CliqueTrees.jl).

# Arguments

  - `algs`: tuple of [elimination algorithms](https://algebraicjulia.github.io/CliqueTrees.jl/stable/api/#Elimination-Algorithms).
  - `level`: maximum level
  - `width`: minimum width
  - `imbalances`: imbalance parameters 

"""
struct HyPar{A} <: Optimizer
    algs::A
    level::Int
    width::Int
    imbalances::StepRange{Int, Int}
end

function HyPar(algs::A = (MF(), MMD());
        level::Integer = 6,
        width::Integer = 120,
        imbalances::AbstractRange=130:130,
    ) where{A}

    return HyPar{A}(algs, level, width, imbalances)
end

function Base.show(io::IO, ::MIME"text/plain", config::HyPar{A}) where {A}
    println(io, "HyPar{$A}:")

    for alg in config.algs
        show(IOContext(io, :indent => 4), "text/plain", alg)
    end

    println(io, "    level: $(config.level)")
    println(io, "    width: $(config.width)")
    println(io, "    imbalances: $(config.imbalances)")
    return
end
