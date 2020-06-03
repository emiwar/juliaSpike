include("main.jl")

#Minimal network
net = JuliaSpike.Network(;dt=0.1f0)
neurons = JuliaSpike.addneurons!(net, :LIF, 2)
JuliaSpike.connect!(net, neurons[1], neurons[2], :Exp)
net_device = JuliaSpike.todevice(net)
JuliaSpike.step!(net_device)
