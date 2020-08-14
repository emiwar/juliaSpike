struct SimulatorVar{T}
    name::Symbol
    default::T
    read::Bool
    write::Bool
    target::Symbol
end

struct NeuronFeature
    stepAction::Expr
    variables::Vector{SimulatorVar}
    order::Float64
end

struct Synapse
    onPre::Expr
    onPost::Expr
    variables::Vector{SimulatorVar}
    preFeatures::Vector{NeuronFeature}
    postFeatures::Vector{NeuronFeature}
    order::Float64
end

struct Population
    features::Vector{NeuronFeature}
    size::Int64
    values::Dict{Symbol, Any}
end
Population(features, size; kwargs...) = Population(features, size, Dict(kwargs))

struct Connection
    prePop::Population
    postPop::Population
    synapse::Synapse
    prePostItr::Any
    values::Dict{Symbol, Any}
end

Connection(prePop, postPop, synapse, prePostItr; kwargs...) = Connection(prePop, postPop, synapse, prePostItr, Dict(kwargs))