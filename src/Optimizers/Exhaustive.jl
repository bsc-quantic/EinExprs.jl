using Base: @kwdef
using Combinatorics
using LinearAlgebra: Symmetric

@doc raw"""
    Exhaustive(; outer = false)

Exhaustive contraction path optimizers. It guarantees to find the optimal contraction path but at a large cost.

# Keywords

- `outer` instructs to consider outer products (aka tensor products) on the search for the optimal contraction path. It rarely provides an advantage over only considering inner products and thus, it is `false` by default.

!!! warning
    The functionality of `outer = true` has not been yet implemented.

# Implementation

The algorithm has a ``\mathcal{O}(n!)`` time complexity if `outer = true` and ``\mathcal{O}(\exp(n))`` if `outer = false`.
"""
@kwdef struct Exhaustive <: Optimizer
    metric::Function = flops
    outer::Bool = false
    strategy::Symbol = :breadth
end

function einexpr(config::Exhaustive, path::SizedEinExpr{L}; cost = BigInt(0)) where {L}
    if config.strategy === :breadth
        return exhaustive_breadthfirst(Val(config.metric), path; outer = config.outer)
    elseif config.strategy === :depth
        init_path = einexpr(Naive(), path)
        leader = Ref((;
            path = init_path,
            cost = mapreduce(config.metric, +, Branches(init_path, inverse = true), init = BigInt(0))::BigInt,
        ))
        exhaustive_depthfirst(Val(config.metric), path, cost, config.outer, leader)
        return leader[].path
    else
        error("Unknown strategy: $(config.strategy)")
    end
end

function exhaustive_depthfirst(
    @specialize(metric::Val{Metric}),
    path::SizedEinExpr{L},
    cost,
    outer,
    leader;
    cache = Dict{Vector{L},BigInt}(),
    hashyperinds = !isempty(hyperinds(path)),
) where {L,Metric}
    if nargs(path) <= 2
        leader[] = (; path = path, cost = cost)
        return
    end

    for (i, j) in combinations(path.args, 2)
        !outer && isdisjoint(head(i), head(j)) && continue
        candidate = sum(i, j; skip = hashyperinds ? path.head ∪ hyperinds(path) : path.head)

        # prune paths based on metric
        new_cost = cost + get!(cache, head(candidate)) do
            Metric(SizedEinExpr(candidate, path.size))
        end
        new_cost >= leader[].cost && continue

        new_path = SizedEinExpr(EinExpr(head(path), [candidate, filter(∉([i, j]), path.args)...]), path.size) # sum([candidate, filter(∉([i, j]), args(path))...], skip = path.head)
        exhaustive_depthfirst(metric, new_path, new_cost, outer, leader; cache, hashyperinds)
    end
end

function exhaustive_breadthfirst(
    @specialize(metric::Val{Metric}),
    expr::SizedEinExpr{L};
    outer::Bool = false,
    hashyperinds = !isempty(hyperinds(expr)),
) where {L,Metric}
    hashyperinds && error("Hyperindices not supported yet")

    cost_fac = maximum(values(expr.size))

    # make a initial guess using a fast optimizer like Greedy
    greedy_path = einexpr(Greedy(), expr)
    cost_max = mapreduce(Metric, +, Branches(greedy_path, inverse = true), init = BigInt(0))::BigInt

    # number of input tensors
    n = nargs(expr)

    # S[c]: set of all objects made up by contracting together `c` unique tensors from S[1]
    # NOTE BitSet contains identifiers (i.e. an `Integer`) of input tensors, so each set is a candidate "contracted" subgraph
    # NOTE it doesn't contain all combinations (as it's combinatorially big); it's filtered by `cost_max`
    S = map(_ -> BitSet[], 1:n)

    # initialize S₁
    S[1] = [sizehint!(BitSet([i]), n) for i in 1:n]

    # caches the best-known cost for constructing each object in S[c]
    # NOTE no cost because no contraction on S₁ (only input tensors)
    costs = Dict{BitSet,BigInt}(s => zero(BigInt) for s in S[1])

    # contains the indices of the intermediate tensors in S
    indices = Dict{BitSet,Vector{L}}(s => head(expr.args[only(s)]) for s in S[1])

    # contains the best-known contraction tree for constructing each object in S[c]
    trees = Dict{BitSet,Tuple{BitSet,BitSet}}(s => (BitSet(), BitSet()) for s in S[1])

    cost_cur = cost_max
    cost_prev = zero(cost_max)

    while cost_cur <= cost_max
        cost_next = cost_max

        # construct all subsets of `c` tensors (S[c]) that fulfill cost <= cost_cur
        for c in 2:n, k in 1:c÷2, (ia, ta) in enumerate(S[k]), (ib, tb) in enumerate(S[c-k])
            # special case for k = c/2 ∈ ℕ (i.e. k == c-k): `S[k] === S[c-k]` and thus, we only need `combinations(S[k], 2)`
            k == c - k && ia >= ib && continue

            # if not disjoint, then ta and tb contain at least one common tensor
            isdisjoint(ta, tb) || continue

            # outer products do not generally improve contraction path
            !outer && isdisjoint(indices[ta], indices[tb]) && continue

            # new candidate contraction
            tc = ta ∪ tb # aka Q in the paper

            # compute cost of getting `tc` by contracting `ta` and `tb
            shallow_expr_a = EinExpr(indices[ta])
            shallow_expr_b = EinExpr(indices[tb])
            expr_c = sum(shallow_expr_a, shallow_expr_b; skip = expr.head)

            μ = costs[ta] + costs[tb] + Metric(SizedEinExpr(expr_c, expr.size))

            # if `μ` is the cheapest known cost for constructing `tc`, record it
            if μ <= get(costs, tc, cost_cur)
                tc ∉ S[c] && push!(S[c], tc)
                costs[tc] = μ
                indices[tc] = head(expr_c)
                trees[tc] = (ta, tb)

            elseif cost_cur < μ < cost_next
                cost_next = μ
            end
        end

        isempty(S[n]) || break

        cost_prev = cost_cur
        cost_cur = min(cost_max, cost_next * cost_fac)
    end

    function recurse_construct(tc)
        ta, tb = trees[tc]

        if isempty(ta) && isempty(tb)
            return EinExpr(indices[tc]::Vector{L})
        end

        return EinExpr(indices[tc], map(recurse_construct, [ta, tb]))
    end

    path = recurse_construct(only(S[n]))
    return SizedEinExpr(path, expr.size)
end
