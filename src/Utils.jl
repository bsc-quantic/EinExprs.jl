using DataStructures: counter

nonunique(v) = [k for (k, v) in counter(v) if v > 1]

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

onehot_intersect(a::T, b::T) where {T<:Integer} = a & b
onehot_intersect(a::BitSet, b::BitSet) = intersect(a, b)

onehot_setdiff(a::T, b::T) where {T<:Integer} = a & ~b
onehot_setdiff(a::BitSet, b::BitSet) = setdiff(a, b)

onehot_symdiff(a::T, b::T) where {T<:Integer} = a ⊻ b
onehot_symdiff(a::BitSet, b::BitSet) = symdiff(a, b)

onehot_only(set::T) where {T<:Integer} = count_ones(set) == 1 ? trailing_zeros(set) + 1 : error("Expected 1 element")
onehot_only(set::BitSet) = only(set)

onehot_isempty(set::T) where {T<:Integer} = set == zero(T)
onehot_isempty(set::BitSet) = isempty(set)

if VERSION >= v"1.9"
    flatmap = Iterators.flatmap
else
    flatmap(f, iterators...) = Iterators.flatten(Iterators.map(f, iterators...))
end

@generated function flatunique(f, itr)
    if Iterators.IteratorEltype(itr) isa Iterators.EltypeUnknown
        return :(flatunique(Any, f, itr))
    end

    fouttype = Base.promote_op(f.instance, eltype(itr))
    if Iterators.IteratorEltype(fouttype) isa Iterators.EltypeUnknown
        return :(flatunique(Any, f, itr))
    end

    return :(flatunique($(eltype(fouttype)), f, itr))
end

function flatunique(::Type{T}, f, itr) where {T}
    u = T[]
    for x in itr
        for y in f(x)
            y ∉ u && push!(u, y)
        end
    end

    return u
end

# TODO move to DelegatorTraits.jl
struct FunctionDelegatable{F} <: Interface end
