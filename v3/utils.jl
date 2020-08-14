import Adapt

struct CuVectorVector{T, I}
    size::CUDA.CuVector{I}
    offset::CUDA.CuVector{I}
    values::CUDA.CuVector{T}
end

function CuVectorVector(a::AbstractArray{T}) where {ST, T <: AbstractArray{ST}}
    sizes = Vector{Int32}(Base.length.(a))
    offsets = Int32[1;cumsum(sizes[1:end-1]).+1]
    values = CUDA.CuVector{ST}(undef, sum(sizes))
    for i=1:Base.length(a)
        if sizes[i] > 0
            values[offsets[i]:offsets[i]+sizes[i]-1] = CUDA.cu(a[i])
        end
    end
    CuVectorVector(CUDA.cu(sizes), CUDA.cu(offsets), values)
end

Base.length(a::CuVectorVector) = Base.length(a.size)
Base.getindex(a::CuVectorVector, i::Int) = view(a.values, a.offset[i]:a.offset[i]+a.size[i]-1)

struct CuDeviceVectorVector{I, T, A1, A2, A3}
    size::CUDA.CuDeviceVector{I, A1}
    offset::CUDA.CuDeviceVector{I, A2}
    values::CUDA.CuDeviceVector{T, A3}
end
Base.length(a::CuDeviceVectorVector) = Base.length(a.size)
Base.getindex(a::CuDeviceVectorVector, i::Int) = view(a.values, a.offset[i]:a.offset[i]+a.size[i]-1)

function Adapt.adapt(to, a::CuVectorVector)
    CuDeviceVectorVector(Adapt.adapt(to, a.size),
                         Adapt.adapt(to, a.offset),
                         Adapt.adapt(to, a.values))
end