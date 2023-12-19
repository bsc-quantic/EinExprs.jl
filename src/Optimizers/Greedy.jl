using Base: @kwdef
using DataStructures: MutableBinaryHeap, update!
using Combinatorics

"""
    Greedy(; metric = removedsize, choose = pop!)

Greedy contraction path solver. Greedily selects contractions that maximize a metric.

# Keywords

  - `metric` is a function that evaluates candidate pairwise tensor contractions. Defaults to [`removedsize`](@ref).
  - `choose` is a function that extracts a pairwise tensor contraction between candidates. Defaults to candidate that maximize `metric` using `pop!`.
  - `outer` If `true`, consider outer products as candidates. Defaults to `false`.

# Implementation

The implementation uses a binary heaptree to sort candidate pairwise tensor contractions. Then recursively,

 1. Selects and extracts a candidate from the heaptree using the `choose` function.
 2. Updates the `metric` of the candidates which contain neighbouring indices to the one selected.
 3. Append the selected index to the path and go back to step 1.
"""
@kwdef struct Greedy <: Optimizer
    metric::Function = removedsize
    choose::Function = pop!
    outer::Bool = false
end

function einexpr(config::Greedy, path, sizedict)
    metric = config.metric(sizedict)

    # generate initial candidate contractions
    queue = MutableBinaryHeap{Tuple{Float64,EinExpr}}(
        Base.By(first, Base.Reverse),
        map(
            Iterators.filter(((a, b),) -> config.outer || !isdisjoint(a.head, b.head), combinations(path.args, 2)),
        ) do (a, b)
            # TODO don't consider outer products
            candidate = sum([a, b], skip = path.head ∪ hyperinds(path))
            weight = metric(candidate)
            (weight, candidate)
        end,
    )

    while nargs(path) > 2 && length(queue) > 1
        # choose winner
        _, winner = config.choose(queue)

        # discard winner if old
        any(∉(args(path)), args(winner)) && continue

        # remove old intermediate tensors
        setdiff!(path.args, args(winner))

        # update candidate queue
        for other in Iterators.filter(other -> config.outer || !isdisjoint(winner.head, other.head), path.args)
            # TODO don't consider outer products
            candidate = sum([winner, other], skip = path.head ∪ hyperinds(path))
            weight = metric(candidate)
            push!(queue, (weight, candidate))
        end

        # append winner to contraction path
        push!(path.args, winner)
    end

    return path
end

function einexpr(config::Greedy, path::SizedEinExpr)
    return einexpr(config, path.path, path.size)
end
