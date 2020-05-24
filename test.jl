include("core.jl")
include("models.jl")

N = 10
connProb = 0.5
net = Network{LIFNeuronParams, LIFNeuronVars, ExpSynapseParams, ExpSynapseVars}(N)
for i=1:N, j=1:N
    if rand() < connProb:
        push!(net.synapses, Synapse(ExpSynapseParams(0.1*rand(), 10.0),
                                    ExpSynapseVars(0.0), i, j))
    end
end

sim = Simulation(net)
