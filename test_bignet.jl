include("main.jl")

#Big network
N = 2750
connProb = 0.5
net = JuliaSpike.Network(;dt=0.1f0)
neurons = JuliaSpike.addneurons!(net, :LIF, N)
JuliaSpike.connectrand!(net, neurons, neurons, :Exp, connProb)

net_device = JuliaSpike.todevice(net)
#JuliaSpike.step!(net_device)
function nSteps!(net, N)
    for i=1:N
        JuliaSpike.step!(net)
    end
end
nSteps!(net_device, 1)
@time nSteps!(net_device, 100000)