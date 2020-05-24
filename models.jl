struct LIFNeuronParams <: NeuronParams
    V_thres::Float32
    V_res::Float32
    tau::Float32
end
struct LIFNeuronVars <: NeuronVars
    V::Float32
end
function dynamics(params::LIFNeuronParams, vars::LIFNeuronVars, I, dt)
    V = (1.0-dt/params.tau)*vars.V + dt*I
    if V > LIFNeuronParams.V_thres
        V = LIFNeuronParams.V_res
    end
    return LIFNeuronVars(V)
end

struct ExpSynapseParams <: SynParams
    weight::Float32
    tau::Float32
end
struct ExpSynapseVars <: SynVars
    I::Float32
end
ExpSynapse = Synapse{ExpSynapseParams, ExpSynapseVars}
function dynamics(params::ExpSynapseParams, vars::ExpSynapseVars, preSpiked, dt)
    I = (1.0-dt/params.tau)*vars.I + preSpiked*params.weight
    return ExpSynapseVars(I)
end

