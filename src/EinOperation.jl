abstract type EinOperation{N} where {N} end

struct AxisPermutation <: EinOperation{1}
    perm::Dict{Symbol,Symbol}

    # AxisPermutation()
end

struct Diagonal <: EinOperation{1}
    inds::NTuple{N,Symbol} where {N}

    Diagonal(inds::Tuple) = new(inds)
end
Diagonal(inds...) = new(inds)

struct AxisSummation <: EinOperation{1}
    inds::NTuple{N,Symbol} where {N}

    AxisSummation(inds::Tuple) = new(inds)
end
AxisSummation(inds...) = AxisSummation(inds)

struct HadamardProduct <: EinOperation{2}
    inds::NTuple{N,Symbol} where {N}

    HadamardProduct(inds::Tuple) = new(inds)
end
HadamardProduct(inds...) = new(inds)

struct OuterProduct <: EinOperation{2} end

struct TensorContraction <: EinOperation{2}
    inds::NTuple{N,Symbol} where {N}

    TensorContraction(inds::Tuple) = new(inds)
end
TensorContraction(inds...) = TensorContraction(inds)
