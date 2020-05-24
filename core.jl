import CuArrays
using CUDAnative

abstract type SynParams end
abstract type SynVars end
struct Synapse{SP, SV}#where {SP<:SynParams, VP<:SynVars}
    params::SP
    vars::SV
    pre::UInt32
    post::UInt32
end

abstract type NeuronParams end
abstract type NeuronVars end
struct Neuron{NP, NV}# where NP <: NeuronParams, <: NeuronVars
    params::NP
    vars::NV
end

struct Network{NP, NV, SP, SV}
    neurons::Vector{Neuron{NP, NV}}
    synapses::Vector{Synapse{SP, SV}}
end

function Network(nNeurons::Int) where {NP, NV, SP, SV}
    Network(fill(Neuron{NP, NV}(NP(), NV()), nNeurons), Vector{Synapse{SP, SV}}())
end

struct Simulation{NP, NV, SP, SV}
    neuronParams::CuArrays.CuVector{NP}
    neuronVars::CuArrays.CuVector{NV}
    neuronSpiked::CuArrays.CuVector{Bool}
    inputCurrent::CuArrays.CuVector{Float32}
    synapseParams::CuArrays.CuVector{SP}
    synapseVars::CuArrays.CuVector{SV}
    pre::CuArrays.CuVector{UInt32}
    post::CuArrays.CuVector{UInt32}
    dt::Float32
end

function Simulation(network, dt)
    order = sortperm(network.synapses, by=s->s.post)
    Simulation(CuArrays.cu(map(n->n.params, network.neurons)),
               CuArrays.cu(map(n->n.vars, network.neurons)),
               CuArrays.fill(false, length(network.neurons)),
               CuArrays.fill(0.0f0, length(network.neurons)),
               CuArrays.cu(map(s->s.params, network.synapses)[order]),
               CuArrays.cu(map(s->s.vars, network.synapses)[order]),
               CuArrays.cu(map(s->s.pre, network.synapses)[order]),
               CuArrays.cu(map(s->s.post, network.synapses)[order]),
               0.1) 
end

function stepNeurons!(simulation::Simulation)
    nBlocks = UInt32(ceil(length(simulation.neurons) / 256.0))
    @cuda blocks=nBlocks threads=256 neuronKernel!(simulation)
end

function neuronKernel!(simulation::Simulation)
    id = blockDim().x * blockIdx().x + threadIdx().x
    if id > length(simulation.neuronParams)
        return nothing
    end
    params = simulation.neuronParams[id]
    vars = simulation.neuronVars[id]
    inputCurrent = simulation.inputCurrent[id]
    newVars, spiked = dynamics(params, vars, inputCurrent)
    simulation.neuronVars[id] = newVars
    neuronSpiked[id] = spiked
    return nothing
end

function stepSynapses!(simulation::Simulation)
    nBlocks = UInt32(ceil(length(simulation.synapseParams) / 256.0))
    @cuda blocks=nBlocks threads=256 synapseKernel!(simulation)
end

function synapseKernel!(simulation::Simulation)
    id = blockDim().x * blockIdx().x + threadIdx().x
    if id > length(simulation.synapseParams)
        return nothing
    end
    
    #Load data
    params = simulation.synpaseParams[id]
    vars = simulation.synpaseVars[id]
    pre = simulation.pre[id]
    post = simulation.post[id]
    preSpiked = simulation.neuronSpiked[pre]
    
    #Simulate synapse
    newVars = dynamics(params, vars, preSpiked, simulation.dt)
    simulation.synpaseVars[id] = newVars
    
    #Gather currents
    postInThread = CUDAnative.@cuStaticSharedMem(UInt32, blockDim().x)
    currInThread = CUDAnative.@cuStaticSharedMem(Float32, blockDim().x)
    postInThread[threadIdx().x] = post
    gatherCurr[threadIdx().x] = newVars.I
    offset = UInt32(1)
    while offset < blockDim().x
        CUDAnative.sync_threads()
        other = threadIdx().x + offset
        if other <= blockDim().x
            if postInThread[other] == post
                gatherCurr[threadIdx().x] += gatherCurr[other]
            end
        end
        offset *= 2
    end
    if threadIdx().x == 1 || postInThread[threadIdx().x - 1] != post
        @atomic simulation.inputCurrent[post] += gatherCurr[threadIdx().x]
    end
    
    return nothing
end