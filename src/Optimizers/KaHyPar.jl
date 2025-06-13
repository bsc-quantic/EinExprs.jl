using Base: @kwdef
using CliqueTrees: MF, MMD

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
