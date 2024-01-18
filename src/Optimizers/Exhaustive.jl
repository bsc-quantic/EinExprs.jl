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
        ninds = length(inds(path))
        settype = if ninds <= 8
            UInt8
        elseif ninds <= 16
            UInt16
        elseif ninds <= 32
            UInt32
        elseif ninds <= 64
            UInt64
        else
            BitSet
        end
        return exhaustive_breadthfirst(Val(config.metric), path, settype; outer = config.outer)
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

onehot_init(T::Type{<:Integer}) = zero(T)
onehot_init(::Type{BitSet}) = BitSet()

function onehot_in(i, set::T) where {T<:Integer}
    i > sizeof(T) * 8 && return false
    mask = one(T) << (i - 1)
    return mask & set != zero(T)
end
onehot_in(i, set::BitSet) = in(i, set)

function onehot_push!(set::T, i) where {T<:Integer}
    i > sizeof(T) * 8 && error("Index out of bounds")
    mask = one(T) << (i - 1)
    set |= mask
    return set
end
onehot_push!(set::BitSet, i) = push!(set, i)

function onehot_pop!(set::T, i) where {T<:Integer}
    i > sizeof(T) * 8 && error("Index out of bounds")
    mask = one(T) << (i - 1)
    set &= ~mask
    return set
end
onehot_pop!(set::BitSet, i) = pop!(set, i)

onehot_isdisjoint(a::T, b::T) where {T<:Integer} = a & b == zero(T)
onehot_isdisjoint(a::BitSet, b::BitSet) = isdisjoint(a, b)

onehot_union(a::T, b::T) where {T<:Integer} = a | b
onehot_union(a::BitSet, b::BitSet) = union(a, b)

onehot_only(set::T) where {T<:Integer} = count_ones(set) == 1 ? trailing_zeros(set) + 1 : error("Expected 1 element")
onehot_only(set::BitSet) = only(set)

onehot_isempty(set::T) where {T<:Integer} = set == zero(T)
onehot_isempty(set::BitSet) = isempty(set)

function exhaustive_breadthfirst(
    @specialize(metric::Val{Metric}),
    expr::SizedEinExpr{L},
    ::Type{SetType} = BitSet;
    outer::Bool = false,
    hashyperinds = !isempty(hyperinds(expr)),
) where {L,Metric,SetType}
    hashyperinds && error("Hyperindices not supported yet")

    cost_fac = maximum(values(expr.size))

    # make a initial guess using a fast optimizer like Greedy
    greedy_path = einexpr(Greedy(), expr)
    cost_max = mapreduce(Metric, +, Branches(greedy_path, inverse = true), init = BigInt(0))::BigInt

    # number of input tensors
    n = nargs(expr)

    # S[c]: set of all objects made up by contracting together `c` unique tensors from S[1]
    # NOTE Set contains identifiers (i.e. an `Integer`) of input tensors, so each set is a candidate "contracted" subgraph
    # NOTE it doesn't contain all combinations (as it's combinatorially big); it's filtered by `cost_max`
    S = map(_ -> SetType[], 1:n)

    # initialize S₁
    S[1] = map(1:n) do i
        onehot_push!(onehot_init(SetType), i)
    end

    # caches the best-known cost for constructing each object in S[c]
    # NOTE no cost because no contraction on S₁ (only input tensors)
    costs = Dict{SetType,BigInt}(s => zero(BigInt) for s in S[1])

    # contains the indices of the intermediate tensors in S
    indices = Dict{SetType,Vector{L}}(s => head(expr.args[onehot_only(s)]) for s in S[1])

    # contains the best-known contraction tree for constructing each object in S[c]
    trees = Dict{SetType,Tuple{SetType,SetType}}(s => (onehot_init(SetType), onehot_init(SetType)) for s in S[1])

    cost_cur = cost_max
    cost_prev = zero(cost_max)

    while cost_cur <= cost_max
        cost_next = cost_max

        # construct all subsets of `c` tensors (S[c]) that fulfill cost <= cost_cur
        for c in 2:n, k in 1:c÷2, (ia, ta) in enumerate(S[k]), (ib, tb) in enumerate(S[c-k])
            # special case for k = c/2 ∈ ℕ (i.e. k == c-k): `S[k] === S[c-k]` and thus, we only need `combinations(S[k], 2)`
            k == c - k && ia >= ib && continue

            # if not disjoint, then ta and tb contain at least one common tensor
            onehot_isdisjoint(ta, tb) || continue

            # outer products do not generally improve contraction path
            !outer && isdisjoint(indices[ta], indices[tb]) && continue

            # new candidate contraction
            tc = onehot_union(ta, tb) # aka Q in the paper
            get(costs, tc, cost_cur) > cost_prev || continue

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

        if onehot_isempty(ta) && onehot_isempty(tb)
            return EinExpr(indices[tc]::Vector{L})
        end

        return EinExpr(indices[tc], map(recurse_construct, [ta, tb]))
    end

    path = recurse_construct(only(S[n]))
    return SizedEinExpr(path, expr.size)
end
