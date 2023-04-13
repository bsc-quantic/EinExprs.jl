using Base: @kwdef
using DataStructures: MutableBinaryHeap, update!

"""
    Greedy

Greedy contraction path solver. Greedily selects contractions that maximize the metric.
"""
@kwdef struct Greedy <: Optimizer
    metric::Function = removedsize
    choose::Function = pop!
end

function einexpr(config::Greedy, expr)
    # generate initial candidate contractions
    queue = MutableBinaryHeap{Tuple{Float64,Vector{Symbol}}}(Base.By(first))

    handles = map(suminds(expr, parallel=true)) do inds
        candidate = sum(select(expr, inds)..., inds=inds)
        weight = config.metric(candidate)
        handle = push!(queue, (weight, inds))
        return sort(inds) => handle
    end |> Dict

    while length(queue) > 1
        # select candidate
        _, winner = config.choose(queue)
        any(∉(suminds(expr)), winner) && continue

        # append winner to contraction path
        neigh = neighbours(expr, winner)

        sum!(expr, winner)

        # update candidate queue
        for inds in filter(inds -> !isdisjoint(neigh, inds), suminds(expr, parallel=true))
            candidate = sum(select(expr, inds)..., inds=inds)
            weight = config.metric(candidate)

            # update involved nodes
            if inds ∈ keys(handles)
                update!(queue, handles[sort(inds)], (weight, inds))
            else
                # if new parallel indices have appeared, delete old nodes and create new ones
                for key in Iterators.map(sort, Iterators.filter(key -> !isdisjoint(inds, key), keys(handles)))
                    # key = sort(key)
                    delete!(queue, handles[key])
                    delete!(handles, key)
                end

                handles[sort(inds)] = push!(queue, (weight, inds))
            end
        end
    end

    return expr
end
