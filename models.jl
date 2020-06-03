constants(::NeuronType{:LIF}) = (tau=20.0f0, V_res=0.0f0, V_thres=1.0f0)
parameters(::NeuronType{:LIF}) = NamedTuple()
variables(::NeuronType{:LIF}) = (V=0.0f0,)
dynamics(::NeuronType{:LIF}; tau, I, V) = -1/tau * V + I
function step(nt::NeuronType{:LIF}; I, dt, V, tau, V_res, V_thres)
    V = V + dt*dynamics(nt; tau=tau, I=I, V=V)
    spiked = V > V_thres
    if spiked
        V = V_res
    end
    return ((V=V,), spiked)
end

constants(::SynapseType{:Exp}) = (tau=10.0f0,)
parameters(::SynapseType{:Exp}) = (weight=0.1f0,)
variables(::SynapseType{:Exp}) = (I=0.0f0,)
dynamics(::SynapseType{:Exp}; tau, I, kwargs...) = -1/tau * I
current(::SynapseType{:Exp}; I, kwargs...) = I
function step(st::SynapseType{:Exp}; prespiked, dt, I, weight, tau)#, kwargs...)
    I = I + dt*dynamics(st; tau=tau, I=I) + weight*Float32(prespiked)
    return (;I=I)
end
