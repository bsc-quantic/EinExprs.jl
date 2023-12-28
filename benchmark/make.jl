using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

push!(LOAD_PATH, "$(@__DIR__)/..")

using BenchmarkTools
using EinExprs

suite = BenchmarkGroup()

suite["naive"] = BenchmarkGroup([])
suite["exhaustive"] = BenchmarkGroup([])
suite["greedy"] = BenchmarkGroup([])
suite["kahypar"] = BenchmarkGroup([])

# BENCHMARK 1
expr = sum([
    EinExpr([:j, :b, :i, :h], Dict(i => 2 for i in [:j, :b, :i, :h])),
    EinExpr([:a, :c, :e, :f], Dict(i => 2 for i in [:a, :c, :e, :f])),
    EinExpr([:j], Dict(i => 2 for i in [:j])),
    EinExpr([:e, :a, :g], Dict(i => 2 for i in [:e, :a, :g])),
    EinExpr([:f, :b], Dict(i => 2 for i in [:f, :b])),
    EinExpr([:i, :h, :d], Dict(i => 2 for i in [:i, :h, :d])),
    EinExpr([:d, :g, :c], Dict(i => 2 for i in [:d, :g, :c])),
])

suite["naive"][1] = @benchmarkable einexpr(EinExprs.Naive(), $expr)
suite["exhaustive"][1] = @benchmarkable einexpr(Exhaustive(), $expr)
suite["greedy"][1] = @benchmarkable einexpr(Greedy(), $expr)
suite["kahypar"][1] = @benchmarkable einexpr(HyPar(), $expr)

# BENCHMARK 2
A = EinExpr([:A, :a, :b, :c], Dict(:A => 2, :a => 2, :b => 2, :c => 2))
B = EinExpr([:b, :d, :e, :f], Dict(:b => 2, :d => 2, :e => 2, :f => 2))
C = EinExpr([:a, :e, :g, :C], Dict(:a => 2, :e => 2, :g => 2, :C => 2))
D = EinExpr([:c, :h, :d, :i], Dict(:c => 2, :h => 2, :d => 2, :i => 2))
E = EinExpr([:f, :i, :g, :j], Dict(:f => 2, :i => 2, :g => 2, :j => 2))
F = EinExpr([:B, :h, :k, :l], Dict(:B => 2, :h => 2, :k => 2, :l => 2))
G = EinExpr([:j, :k, :l, :D], Dict(:j => 2, :k => 2, :l => 2, :D => 2))
expr = sum([A, B, C, D, E, F, G], skip = [:A, :B, :C, :D])

suite["naive"][2] = @benchmarkable einexpr(EinExprs.Naive(), $expr)
suite["exhaustive"][2] = @benchmarkable einexpr(Exhaustive(), $expr)
suite["greedy"][2] = @benchmarkable einexpr(Greedy(), $expr)
suite["kahypar"][2] = @benchmarkable einexpr(HyPar(), $expr)

# Tuning
tune!(suite)

# Run
GC.enable(false)
results = run(suite, verbose = true)
GC.enable(true)

using BenchmarkPlots, StatsPlots

for (method, group) in results
    plt = plot(
        results[method],
        yaxis = (:log10, "Execution time [ns]"),
        xaxis = (:flip, "Benchmark set"),
        title = "$method",
    )
    display(plt)
end
