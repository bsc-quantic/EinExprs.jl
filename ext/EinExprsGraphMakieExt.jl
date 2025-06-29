module EinExprsGraphMakieExt

using EinExprs
using EinExprs: Branches
using Graphs
using Makie
using GraphMakie
using AbstractTrees

# NOTE this is a hack! removes NetworkLayout dependency but can be unstable
__networklayout_dim(x) = supertype(typeof(x)).parameters |> first

# TODO rework size calculating algorithm
const MAX_EDGE_WIDTH = 10.0
const MAX_ARROW_SIZE = 35.0
const MAX_NODE_SIZE = 40.0

function Makie.plot(path::SizedEinExpr; kwargs...)
    f = Figure()
    ax, p = plot!(f[1, 1], path; kwargs...)
    return Makie.FigureAxisPlot(f, ax, p)
end

function Makie.plot!(f::Union{Figure,GridPosition}, path::SizedEinExpr; kwargs...)
    ax = if haskey(kwargs, :layout) && __networklayout_dim(kwargs[:layout]) == 3
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
    Colorbar(
        f[1, 2],
        get_edge_plot(p);
        label = "SIZE",
        flipaxis = true,
        flip_vertical_label = true,
        labelsize = 24,
        height = Relative(5 // 6),
        scale = log2,
    )

    Colorbar(
        f[1, 0],
        get_node_plot(p);
        label = "FLOPS",
        flipaxis = false,
        labelsize = 24,
        height = Relative(5 // 6),
        scale = log10,
    )

    return Makie.AxisPlot(ax, p)
end

# TODO replace `to_colormap(:viridis)[begin:end-10]` with a custom colormap
function Makie.plot!(
    ax::Union{Axis,Axis3},
    path::SizedEinExpr;
    colormap = to_colormap(:viridis)[begin:(end-10)],
    inds = false,
    kwargs...,
)
    handles = IdDict(obj => i for (i, obj) in enumerate(PostOrderDFS(path.path)))
    graph = SimpleDiGraph([Edge(handles[from], handles[to]) for to in Branches(path.path) for from in to.args])

    lin_size = length.(PostOrderDFS(path))[1:(end-1)]
    lin_flops = map(max, Iterators.repeated(1), Iterators.map(flops, PostOrderDFS(path)))

    log_size = log2.(lin_size)
    log_flops = log10.(lin_flops)

    kwargs = Dict{Symbol,Any}(kwargs)

    # configure graphics
    get!(kwargs, :edge_width) do
        map(log_size ./ maximum(log_size) .* MAX_EDGE_WIDTH) do x
            iszero(x) ? 4.0 : x
        end
    end

    get!(kwargs, :arrow_size) do
        map(log_size ./ maximum(log_size) .* MAX_ARROW_SIZE) do x
            iszero(x) ? 30.0 : x
        end
    end

    get!(() -> log_flops ./ maximum(log_flops) .* MAX_NODE_SIZE, kwargs, :node_size)

    get!(kwargs, :edge_color, lin_size)
    get!(kwargs, :node_color, lin_flops)

    get!(
        kwargs,
        :arrow_attr,
        (
            colorrange = extrema(lin_size),
            colormap = colormap,
            colorscale = log2,
            highclip = Makie.Automatic(),
            lowclip = Makie.Automatic(),
        ),
    )
    get!(
        kwargs,
        :edge_attr,
        (
            colorrange = extrema(lin_size),
            colormap = colormap,
            colorscale = log2,
            highclip = Makie.Automatic(),
            lowclip = Makie.Automatic(),
        ),
    )
    # TODO replace `to_colormap(:plasma)[begin:end-50]), kwargs...)` with a custom colormap
    get!(
        kwargs,
        :node_attr,
        (
            colorrange = extrema(lin_flops),
            colormap = to_colormap(:plasma)[begin:(end-50)],
            colorscale = log10,
            highclip = Makie.Automatic(),
            lowclip = Makie.Automatic(),
        ),
    )

    # configure labels
    inds == true && get!(() -> join.(head.(PostOrderDFS(path)))[1:(end-1)], kwargs, :elabels)
    get!(() -> repeat([:black], ne(graph)), kwargs, :elabels_color)
    get!(() -> log_size ./ maximum(log_size) .* 5 .+ 12, kwargs, :elabels_textsize)

    # plot graph
    graphplot!(ax, graph; kwargs...)
end

end
