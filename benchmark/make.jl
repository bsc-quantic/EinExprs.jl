using BenchmarkTools
using CairoMakie
using CliqueTrees
using EinExprs
using Graphs
using Random

import KaHyPar

Random.seed!(1)

function random_regular_eincode(n::Integer, k::Integer)
    graph = random_regular_graph(n, k)
    exprs = Vector{EinExpr{Int}}(undef, n)
    sizedict = Dict{Int, Int}()
    
    for v in vertices(graph)
        exprs[v] = EinExpr(Int[])
    end
    
    for (i, edge) in enumerate(edges(graph))
        v = src(edge)
        w = dst(edge)
        push!(head(exprs[v]), i)
        push!(head(exprs[w]), i)
        sizedict[i] = 2
    end
    
    return SizedEinExpr(sum(exprs), sizedict)
end

function make()
    # construct ein-expressions
    exprs = map(3:6) do k
        random_regular_eincode(16, k)
    end

    # construct benchmarks
    n = length(exprs)

    suite = BenchmarkGroup()
    suite["exhaustive"] = BenchmarkGroup([])
    suite["greedy"] = BenchmarkGroup([])
    suite["kahypar"] = BenchmarkGroup([])
    suite["min-fill"] = BenchmarkGroup([])

    count = Dict{String, Vector{Int}}()
    count["exhaustive"] = Vector{Int}(undef, n)
    count["greedy"] = Vector{Int}(undef, n)
    count["kahypar"] = Vector{Int}(undef, n)
    count["min-fill"] = Vector{Int}(undef, n)

    for i in 1:n
        expr = exprs[i]

        suite["exhaustive"][i] = @benchmarkable einexpr(Exhaustive(), $expr)
        suite["greedy"][i] = @benchmarkable einexpr(Greedy(), $expr)
        suite["kahypar"][i] = @benchmarkable einexpr(HyPar(), $expr)
        suite["min-fill"][i] = @benchmarkable einexpr(LineGraph(MF()), $expr)

        count["exhaustive"][i] = mapreduce(flops, +, Branches(einexpr(Exhaustive(), expr)))
        count["greedy"][i] = mapreduce(flops, +, Branches(einexpr(Greedy(), expr)))
        count["kahypar"][i] = mapreduce(flops, +, Branches(einexpr(HyPar(), expr)))
        count["min-fill"][i] = mapreduce(flops, +, Branches(einexpr(LineGraph(MF()), expr)))
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
