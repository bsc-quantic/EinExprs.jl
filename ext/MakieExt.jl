module MakieExt

if isdefined(Base, :get_extension)
    using EinExprs
else
    using ..EinExprs
end

using Graphs
using Makie
using GraphMakie
using NetworkLayout: dim

# TODO rework size calculating algorithm
const MAX_EDGE_WIDTH = 10.0
const MAX_ARROW_SIZE = 35.0
const MAX_NODE_SIZE = 40.0


function Makie.plot(path::EinExpr; kwargs...)
    f = Figure()
    ax, p = plot!(f[1, 1], path; kwargs...)
    return Makie.FigureAxisPlot(f, ax, p)
end

function Makie.plot!(f::Union{Figure,GridPosition}, path::EinExpr; kwargs...)
    ax = if haskey(kwargs, :layout) && dim(kwargs[:layout]) == 3
        Axis3(f[1, 1])
    else
        ax = Axis(f[1, 1])
        ax.aspect = DataAspect()
        ax
    end

    hidedecorations!(ax)
    hidespines!(ax)

    p = plot!(ax, path; kwargs...)

    # plot colorbars
    # TODO configurable `labelsize`
    # TODO configurable alignments
    size_bar = Colorbar(f[1, 2], get_edge_plot(p), label=L"\log_{2}(size)", flipaxis=true, flip_vertical_label=true, labelsize=34)
    size_bar.height = Relative(5 / 6)

    flops_bar = Colorbar(f[1, 0], get_node_plot(p), label=L"\log_{10}(flops)", flipaxis=false, labelsize=34)
    flops_bar.height = Relative(5 / 6)

    return Makie.AxisPlot(ax, p)
end

# TODO replace `to_colormap(:viridis)[begin:end-10]` with a custom colormap
function Makie.plot!(ax::Union{Axis,Axis3}, path::EinExpr; colormap=to_colormap(:viridis)[begin:end-10], labels=false, kwargs...)
    handles = IdDict(obj => i for (i, obj) in enumerate(path))
    graph = SimpleDiGraph([
        Edge(handles[from], handles[to])
        for to in Iterators.filter(obj -> obj isa EinExpr, path)
        for from in to.args
    ])

    log_size = log2.(length.(path))[1:end-1]
    log_flops = log10.(max.((1.0,), flops.(path)))

    min_size, max_size = extrema(log_size)
    min_flops, max_flops = extrema(log_flops)

    kwargs = Dict{Symbol,Any}(kwargs)

    # configure graphics
    get!(() -> log_size ./ max_size .* MAX_EDGE_WIDTH, kwargs, :edge_width)
    get!(() -> log_size ./ max_size .* MAX_ARROW_SIZE, kwargs, :arrow_size)
    get!(() -> log_flops ./ max_flops .* MAX_NODE_SIZE, kwargs, :node_size)

    get!(kwargs, :edge_color, log_size)
    get!(kwargs, :node_color, log_flops)

    get!(kwargs, :arrow_attr, (colorrange=(min_size, max_size), colormap=colormap))
    get!(kwargs, :edge_attr, (colorrange=(min_size, max_size), colormap=colormap))
    # TODO replace `to_colormap(:plasma)[begin:end-50]), kwargs...)` with a custom colormap
    get!(kwargs, :node_attr, (colorrange=(min_flops, max_flops), colormap=to_colormap(:plasma)[begin:end-50]))

    labels == true && get!(() -> join.(EinExprs.labels.(path))[1:end-1], kwargs, :elabels)
    get!(() -> repeat([:black], ne(graph)), kwargs, :elabels_color)
    get!(() -> log_size ./ max_size .* 5 .+ 12, kwargs, :elabels_textsize)

    # plot graph
    graphplot!(ax, graph; kwargs...)
end

end