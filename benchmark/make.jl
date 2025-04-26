using BenchmarkTools
using CairoMakie
using CliqueTrees
using EinExprs

import KaHyPar
import TreeWidthSolver

function make()
    exprs = SizedEinExpr{Symbol}[]

    # benchmark 1
    expr = sum([
        EinExpr([:j, :b, :i, :h], Dict(i => 2 for i in [:j, :b, :i, :h])),
        EinExpr([:a, :c, :e, :f], Dict(i => 2 for i in [:a, :c, :e, :f])),
        EinExpr([:j], Dict(i => 2 for i in [:j])),
        EinExpr([:e, :a, :g], Dict(i => 2 for i in [:e, :a, :g])),
        EinExpr([:f, :b], Dict(i => 2 for i in [:f, :b])),
        EinExpr([:i, :h, :d], Dict(i => 2 for i in [:i, :h, :d])),
        EinExpr([:d, :g, :c], Dict(i => 2 for i in [:d, :g, :c])),
    ])

    push!(exprs, expr)

    # benchmark 2
    A = EinExpr([:A, :a, :b, :c], Dict(:A => 2, :a => 2, :b => 2, :c => 2))
    B = EinExpr([:b, :d, :e, :f], Dict(:b => 2, :d => 2, :e => 2, :f => 2))
    C = EinExpr([:a, :e, :g, :C], Dict(:a => 2, :e => 2, :g => 2, :C => 2))
    D = EinExpr([:c, :h, :d, :i], Dict(:c => 2, :h => 2, :d => 2, :i => 2))
    E = EinExpr([:f, :i, :g, :j], Dict(:f => 2, :i => 2, :g => 2, :j => 2))
    F = EinExpr([:B, :h, :k, :l], Dict(:B => 2, :h => 2, :k => 2, :l => 2))
    G = EinExpr([:j, :k, :l, :D], Dict(:j => 2, :k => 2, :l => 2, :D => 2))
    expr = sum([A, B, C, D, E, F, G], skip = [:A, :B, :C, :D])
    push!(exprs, expr)

    # construct benchmarks
    n = length(exprs)

    suite = BenchmarkGroup()
    suite["naive"] = BenchmarkGroup([])
    suite["exhaustive"] = BenchmarkGroup([])
    suite["greedy"] = BenchmarkGroup([])
    suite["kahypar"] = BenchmarkGroup([])
    suite["min-fill"] = BenchmarkGroup([])
    suite["bouchitte-todinca"] = BenchmarkGroup([])

    count = Dict{String, Vector{Int}}()
    count["naive"] = Vector{Int}(undef, n)
    count["exhaustive"] = Vector{Int}(undef, n)
    count["greedy"] = Vector{Int}(undef, n)
    count["kahypar"] = Vector{Int}(undef, n)
    count["min-fill"] = Vector{Int}(undef, n)
    count["bouchitte-todinca"] = Vector{Int}(undef, n)

    for i in 1:n
        expr = exprs[i]

        suite["naive"][i] = @benchmarkable einexpr(EinExprs.Naive(), $expr)
        suite["exhaustive"][i] = @benchmarkable einexpr(Exhaustive(), $expr)
        suite["greedy"][i] = @benchmarkable einexpr(Greedy(), $expr)
        suite["kahypar"][i] = @benchmarkable einexpr(HyPar(), $expr)
        suite["min-fill"][i] = @benchmarkable einexpr(LineGraph(MF()), $expr)
        suite["bouchitte-todinca"][i] = @benchmarkable einexpr(LineGraph(SafeRules(BT())), $expr)

        count["naive"][i] = mapreduce(flops, +, Branches(einexpr(EinExprs.Naive(), expr)))
        count["exhaustive"][i] = mapreduce(flops, +, Branches(einexpr(Exhaustive(), expr)))
        count["greedy"][i] = mapreduce(flops, +, Branches(einexpr(Greedy(), expr)))
        count["kahypar"][i] = mapreduce(flops, +, Branches(einexpr(HyPar(), expr)))
        count["min-fill"][i] = mapreduce(flops, +, Branches(einexpr(LineGraph(MF()), expr)))
        count["bouchitte-todinca"][i] = mapreduce(flops, +, Branches(einexpr(LineGraph(SafeRules(BT())), expr)))
    end

    # tune benchmarks
    tune!(suite)

    # run benchmarks
    results = run(suite, verbose = true)

    # construct plots
    names = String[]
    x = Vector{Float64}[]
    y = Vector{Float64}[]

    for i in 1:n
        push!(x, Float64[])
        push!(y, Float64[])
    end
        
    for name in keys(count)
        push!(names, name)
        
        for i in 1:n
            push!(x[i], count[name][i])
            push!(y[i], time(minimum(results[name][i])))
        end
    end

    figure = Figure()

    for i in 1:n
        axis = Axis(figure[i, 1];
            xlabel = "flops",
            ylabel = "time (ns)",
            title = "benchmark $i",
            xscale=log10,
            yscale=log10,
            xautolimitmargin = (0.1, 0.2),
            yautolimitmargin = (0.1, 0.2),
        )

        scatter!(axis, x[i], y[i])
        text!(axis, x[i], y[i]; text=names)
    end

    save("figure.png", figure)
end

make()
