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

# m: number of trials
# n: number of vertices
# k: number of neighbors (3, 4, ..., 2 + k)
function make(m::Integer, n::Integer, k::Integer)
    # construct ein-expressions
    exprs = Matrix{SizedEinExpr{Int}}(undef, k, m)

    for i in 1:k, j in 1:m
        exprs[i, j] = random_regular_eincode(n, 2 + i)
    end

    # construct benchmarks
    suite = BenchmarkGroup()
    count = Dict{String, Matrix{Int}}()
    count["exhaustive"] = Matrix{Int}(undef, k, m)
    count["greedy"] = Matrix{Int}(undef, k, m)
    count["kahypar"] = Matrix{Int}(undef, k, m)
    count["min-fill"] = Matrix{Int}(undef, k, m)

    for i in 1:k, j in 1:m
        expr = exprs[i, j]

        suite["exhaustive"][i, j] = @benchmarkable einexpr(Exhaustive(), $expr)
        suite["greedy"][i, j] = @benchmarkable einexpr(Greedy(), $expr)
        suite["kahypar"][i, j] = @benchmarkable einexpr(HyPar(), $expr)
        suite["min-fill"][i, j] = @benchmarkable einexpr(LineGraph(MF()), $expr)

        count["exhaustive"][i, j] = mapreduce(flops, +, Branches(einexpr(Exhaustive(), expr)))
        count["greedy"][i, j] = mapreduce(flops, +, Branches(einexpr(Greedy(), expr)))
        count["kahypar"][i, j] = mapreduce(flops, +, Branches(einexpr(HyPar(), expr)))
        count["min-fill"][i, j] = mapreduce(flops, +, Branches(einexpr(LineGraph(MF()), expr)))
    end

    # tune benchmarks
    tune!(suite; verbose=true)

    # run benchmarks
    results = run(suite, verbose = true)

    # construct plots
    names = String[]
    x = Vector{Float64}[]
    y = Vector{Float64}[]

    for i in 1:k
        push!(x, Float64[])
        push!(y, Float64[])
    end
        
    for name in keys(count)
        push!(names, name)
        
        for i in 1:k
            xx = 0.0
            yy = 0.0

            for j in 1:m
                xx += count[name][i, j] / m
                yy += time(minimum(results[name][i, j])) / m
            end
    
            push!(x[i], xx)
            push!(y[i], yy)
        end
    end

    figure = Figure(; size=(600, 200 * k))

    for i in 1:k - 1
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

    axis = Axis(figure[k, 1];
        ylabel = "time (ns)",
        xlabel = "flops",
        xscale=log10,
        yscale=log10,
        xautolimitmargin = (0.1, 0.2),
        yautolimitmargin = (0.1, 0.2),
    )

    scatter!(axis, x[k], y[k])
    text!(axis, x[k], y[k]; text=names)

    save("figure.png", figure)
end

make(5, 16, 3)
