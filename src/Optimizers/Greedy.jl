using Base: @kwdef
using DataStructures: MutableBinaryHeap, update!
using Combinatorics

"""
    Greedy(; metric = removedsize, choose = pop!)

Greedy contraction path solver. Greedily selects contractions that maximize a metric.

# Keywords

  - `metric` is a function that evaluates candidate pairwise tensor contractions. Defaults to [`removedsize`](@ref).
  - `choose` is a function that extracts a pairwise tensor contraction between candidates. Defaults to candidate that maximize `metric` using `pop!`.

# Implementation

The implementation uses a binary heaptree to sort candidate pairwise tensor contractions. Then recursively,

 1. Selects and extracts a candidate from the heaptree using the `choose` function.
 2. Updates the `metric` of the candidates which contain neighbouring indices to the one selected.
 3. Append the selected index to the path and go back to step 1.
"""
@kwdef struct Greedy <: Optimizer
    metric::Function = removedsize
    choose::Function = pop!
end

function einexpr(config::Greedy, path)
    # generate initial candidate contractions
    queue = MutableBinaryHeap{Tuple{Float64,EinExpr}}(
        Base.By(first, Base.Reverse),
        map(combinations(path.args, 2)) do (a, b)
            # TODO don't consider outer products
            candidate = sum([a, b]) # TODO don't sum output inds
            weight = config.metric(candidate)
            (weight, candidate)
        end,
    )

    while length(path.args) > 2 && length(queue) > 1
        # choose winner
        _, winner = config.choose(queue)

        # discard winner if old
        any(∉(args(path)), args(winner)) && continue

        # remove old intermediate tensors
        setdiff!(path.args, args(winner))

        # update candidate queue
        for other in path.args
            # TODO don't consider outer products
            candidate = sum([winner, other]) # TODO don't sum output inds
            weight = config.metric(candidate)
            push!(queue, (weight, candidate))
        end

        # append winner to contraction path
        push!(path.args, winner)
    end

    return path
end
