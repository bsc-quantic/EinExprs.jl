using BenchmarkTools
using CairoMakie
using CliqueTrees
using EinExprs

import KaHyPar
import TreeWidthSolver

function make()
    # benchmark 1
    expr1 = SizedEinExpr(
        sum([
            EinExpr([1]),
            EinExpr([1, 2]),
            EinExpr([3]),
            EinExpr([3, 4]),
            EinExpr([3, 5]),
            EinExpr([2, 4, 6]),
            EinExpr([6, 7]),
            EinExpr([5, 6, 8]),
        ]),
        Dict(1 => 2, 2 => 2, 3 => 2, 4 => 2, 5 => 2, 6 => 2, 7 => 2, 8 => 2),
    )

    # benchmark 2
    expr2 = SizedEinExpr(
        sum([
            EinExpr([4]),
            EinExpr([1, 2]),
            EinExpr([3, 5, 6]),
            EinExpr([2, 6, 8]),
            EinExpr([1, 3]),
            EinExpr([1]),
            EinExpr([4, 5]),
            EinExpr([6, 7]),
        ]),
        Dict(1 => 2, 2 => 2, 3 => 2, 4 => 2, 5 => 2, 6 => 2, 7 => 2, 8 => 2),
    )

    # benchmark 3
    expr3 = SizedEinExpr(
        sum([
            EinExpr([1]),
            EinExpr([1, 2]),
            EinExpr([3]),
            EinExpr([1, 3, 4]),
            EinExpr([4, 5]),
            EinExpr([1, 3, 6]),
            EinExpr([6, 7]),
            EinExpr([8]),
            EinExpr([3, 8, 9]),
            EinExpr([9, 10]),
        ]),
        Dict(1 => 2, 2 => 2, 3 => 2, 4 => 2, 5 => 2, 6 => 2, 7 => 2, 8 => 2, 9 => 2, 10 => 2),
    )

    # benchmark 4
    expr4 = SizedEinExpr(
        sum([
            EinExpr([1, 2, 8]),
            EinExpr([2, 4, 8]),
            EinExpr([3, 8, 9]),
            EinExpr([4, 8, 9, 11]),
            EinExpr([5, 8, 9]),
            EinExpr([6, 7, 10]),
            EinExpr([7, 10]),
            EinExpr([8, 9]),
            EinExpr([9]),
            EinExpr([10]),
            EinExpr([9, 8, 11])
        ]),
        Dict(1 => 2, 2 => 2, 3 => 2, 4 => 2, 5 => 2, 6 => 2, 7 => 2, 8 => 2, 9 => 2, 10 => 2, 11 => 2),
    )

    # construct benchmarks
    exprs = [expr1, expr2, expr3, expr4]
    n = length(exprs)

    suite = BenchmarkGroup()
    suite["exhaustive"] = BenchmarkGroup([])
    suite["greedy"] = BenchmarkGroup([])
    suite["kahypar"] = BenchmarkGroup([])
    suite["min-fill"] = BenchmarkGroup([])
    suite["bouchitte-todinca"] = BenchmarkGroup([])

    count = Dict{String, Vector{Int}}()
    count["exhaustive"] = Vector{Int}(undef, n)
    count["greedy"] = Vector{Int}(undef, n)
    count["kahypar"] = Vector{Int}(undef, n)
    count["min-fill"] = Vector{Int}(undef, n)
    count["bouchitte-todinca"] = Vector{Int}(undef, n)

    for i in 1:n
        expr = exprs[i]

        suite["exhaustive"][i] = @benchmarkable einexpr(Exhaustive(; strategy=:depth), $expr)
        suite["greedy"][i] = @benchmarkable einexpr(Greedy(), $expr)
        suite["kahypar"][i] = @benchmarkable einexpr(HyPar(), $expr)
        suite["min-fill"][i] = @benchmarkable einexpr(LineGraph(MF()), $expr)
        suite["bouchitte-todinca"][i] = @benchmarkable einexpr(LineGraph(SafeRules(BT())), $expr)

        count["exhaustive"][i] = mapreduce(flops, +, Branches(einexpr(Exhaustive(; strategy=:depth), expr)))
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

    figure = Figure(; size=(600, 800))

    for i in 1:n - 1
        axis = Axis(figure[i, 1];
            ylabel = "time (ns)",
            xscale=log10,
            yscale=log10,
            xautolimitmargin = (0.1, 0.2),
            yautolimitmargin = (0.1, 0.2),
        )

        scatter!(axis, x[i], y[i])
        text!(axis, x[i], y[i]; text=names)
    end

    axis = Axis(figure[n, 1];
        ylabel = "time (ns)",
        xlabel = "flops",
        xscale=log10,
        yscale=log10,
        xautolimitmargin = (0.1, 0.2),
        yautolimitmargin = (0.1, 0.2),
    )

    scatter!(axis, x[n], y[n])
    text!(axis, x[n], y[n]; text=names)

    save("figure.png", figure)
end

make()
