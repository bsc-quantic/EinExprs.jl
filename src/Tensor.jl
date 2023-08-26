using ImmutableArrays

const Tensor = @NamedTuple{array::AbstractArray, inds::ImmutableVector{Symbol,Vector{Symbol}}}

function Tensor(array, inds)
    ndims(array) == length(inds) ||
        throw(ArgumentError("ndims(array) [$(ndims(array))] and length(inds) [$(length(inds))] must match"))

    Tensor((array, inds))
end

head(tensor::Tensor) = tensor.inds
args(::Tensor) = []
inds(tensor::Tensor) = head(tensor)

Base.size(tensor::Tensor) = size(tensor.array)
Base.size(tensor::Tensor, i) = size(tensor.array, i)
Base.size(tensor::Tensor, index::Symbol) = size(tensor, findfirst(==(index), inds(tensor)))

Base.selectdim(tensor::Tensor, d::Integer, i) = Tensor(selectdim(tensor.array, d, i), inds(tensor))
function Base.selectdim(tensor::Tensor, d::Integer, i::Integer)
    data = selectdim(tensor.array, d, i)
    indices = [index for (i, index) in enumerate(inds(tensor)) if i != d]
    Tensor(data, indices)
end

Base.selectdim(tensor::Tensor, d::Symbol, i) = selectdim(tensor, findfirst(==(d), tensor.inds), i)
