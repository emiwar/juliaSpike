import CuArrays
import CUDAnative

abstract type Synapse end
abstract type Neuron end

struct ExpSynapse <: Synapse
    pre::Int32
    post::Int32
    weight::Float32
    I::Float32
end
ExpSynapse(pre, post, weight) = ExpSynapse(pre, post, weight, 0.0)
function dynamics(syn::ExpSynapse, sim, dt)
    preSpiked = sim.neurons[syn.pre].spiked
    tau = 10.0 #ms
    I = syn.I - dt/tau * syn.I + preSpiked*syn.weight
    ExpSynapse(syn.pre, syn.post, syn.weight, I)
end

struct LIFNeuron <: Neuron
    I::Float32
    V::Float32
    spiked::Bool
end
LIFNeuron() = LIFNeuron(0.0, 0.0)

struct Network{NeuronType <: Neuron, SynapseType <: Synapse}
    neurons::Vector{NeuronType}
    synapses::Vector{SynapseType}
end

function Network{NeuronType, SynapseType}(nNeurons::Int) where {NeuronType <: Neuron, SynapseType <: Synapse}
    Network(fill(NeuronType(), nNeurons), Vector{SynapseType}())
end

function connect!(network::Network, pre::Int32, post::Int32, weight::Float32)
    push!(network.synapses, ExpSynapse(pre, post, weight))
end

struct Simulation{NeuronType <: Neuron, SynapseType <: Synapse}
    neurons::CuArrays.CuVector{NeuronType}
    synapses::CuArrays.CuVector{SynapseType}
end

function Simulation(network)
    sort!(network.synapses, by=(s->s.post))
    Simulation(CuArrays.cu(network.neurons), CuArrays.cu(network.synapses))
end

function stepSynapses!(simulation::Simulation)
    nBlocks = Int32(ceil(length(simulation.synpases) / 256.0))
    
end

function synapseKernel!(simulation::Simulation, dt::Float32)
    synapseId = blockDim().x * blockIdx().x + threadIdx().x
    synapse = simulation.synpases[synapseId]
    simulation.synpases[synapseId] = dynamics(synapse, simulation, dt)
end

### Test ###
N = 5000
net = Network(N)
for i=1:N, j=1:N
    if rand() < connProb:
        connect!(net, i, j, 0.1*rand())

sim = Simulation(net)
