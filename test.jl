#using Revise
include("main.jl")
#using JuliaSpike

#Minimal network
net = JuliaSpike.Network(;dt=0.1f0)
neurons = [JuliaSpike.addneuron!(net, :LIF) for i=1:2]
JuliaSpike.connect!(net, neurons[1], neurons[2], :Exp)
net_device = JuliaSpike.todevice(net)
JuliaSpike.step!(net_device)

#One spike
net_device = JuliaSpike.todevice(net)
for t=1:10
    JuliaSpike.step!(net_device)
    if t==5
        net_device.neurongroups[:LIF].spiked[1] = true
    end
    #println(net_device.synapsegroups[:Exp, :LIF, :LIF].vars)
    #println(net_device.neurongroups[:LIF].inputCurrent)
    #println(net_device.neurongroups[:LIF].vars)
end

#Big network
N = 2000
connProb = 0.5
net = JuliaSpike.Network(;dt=0.1f0)
neurons = JuliaSpike.addneurons!(net, :LIF, N)
#[JuliaSpike.addneuron!(net, :LIF) for i=1:N]
JuliaSpike.connectrand!(net, neurons, neurons, :Exp, connProb)
#for i=1:N, j=1:N
#    if rand() < connProb
#        JuliaSpike.connect!(net, neurons[i], neurons[j], :Exp)
#    end
#end

net_device = JuliaSpike.todevice(net)
JuliaSpike.step!(net_device)
function nSteps!(net, N)
    for i=1:N
        JuliaSpike.step!(net)
    end
end
@time nSteps!(net_device, 10000)
