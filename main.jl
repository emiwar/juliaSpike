module JuliaSpike
using CUDA
using Adapt

include("core.jl")
include("models.jl")
include("utils.jl")
#include("gpu.jl")

export Network, NeuronType, SynapseType, todevice, addneuron!, connect!, step!
end