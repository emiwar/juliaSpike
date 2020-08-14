import CUDA
include("utils.jl")
include("network.jl")
include("build.jl")
include("run.jl")

LIF = NeuronFeature(
    quote
        V -= dt/tau * V
        spiked = V>1
        V *= !spiked
    end,
    [SimulatorVar(:V, 0.0f0, true, true, :local),
     SimulatorVar(:spiked, false, false, true, :local),
     SimulatorVar(:tau, 10.0f0, true, false, :global),
     SimulatorVar(:dt, 0.5f0, true, false, :global)],
    0.0)

#Spiking = NeuronFeature(:(), [SimulatorVar(:spiked, false, false, true, :local)], 0.0)
#DeltaPost = NeuronFeature(:(), [SimulatorVar(:V, false, true, true, :local)], 0.0)
DeltaSynapse = Synapse(:(post_V += W), :(),
    [SimulatorVar(:W, 0.1f0, true, false, :local),
     SimulatorVar(:spiked, false, false, false, :pre),
     SimulatorVar(:V, 0.0f0, true, true, :post)],
    [], [], 0.0)

pop = Population([LIF], 10)
conn = Connection(pop, pop, DeltaSynapse, [(1,2), (2,3), (3,4), (1, 5)])
net = Network([pop], [conn], :gpu)

traces = Vector{Float32}[]

for t=1:100
    t==3 && (net.populations[1].localVars.V[1] = 1.1)
    step!(net)
    push!(traces, Vector(net.populations[1].localVars.V))
end

#bc = net.connections[1]
#CUDA.@device_code bc.onPre(1, bc.indexMap.preToSyn[1], bc.indexMap.synToPost, bc.localVars, bc.preVars, bc.postVars, net.globalVars)