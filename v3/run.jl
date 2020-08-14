function step!(network::Network)
    #Neurons
    for population in network.populations
        N = length(population)
        threads = min(N, 256)
        blocks = (N + 255) ÷ 256
        @CUDA.cuda blocks=blocks threads=threads population.updateFcn(population.localVars, network.globalVars)
    end
    CUDA.synchronize()

    #Synapses
    for connection in network.connections
        #On pre
        if haskey(connection.preVars, :spiked)
            N = Base.length(connection.indexMap.preToSyn)
            threads = min(N, 256)
            blocks = (N + 255) ÷ 256
            @CUDA.cuda blocks=blocks threads=threads handleSpikes!(
                    connection.preVars.spiked, connection.onPre,
                    connection.indexMap.preToSyn, connection.indexMap.synToPost,
                    connection.localVars, connection.preVars,
                    connection.postVars, network.globalVars)
        end
        
        #On post
        if haskey(connection.postVars, :spiked)
            N = length(connection.indexMap.postToSyn)
            threads = min(N, 256)
            blocks = (N + 255) ÷ 256
            @CUDA.cuda blocks=blocks threads=threads handleSpikes!(
                    connection.postVars.spiked, connection.onPost,
                    connection.indexMap.postToSyn, connection.indexMap.synToPre,
                    connection.localVars, connection.preVars,
                    connection.postVars, network.globalVars)
        end
    end
    CUDA.synchronize()
    return nothing
end

function handleSpikes!(spiked, fcn, siteToSyn, synToTarget,
                       localVars, preVars, postVars, globalVars)
    site_idx = CUDA.blockDim().x*(CUDA.blockIdx().x-1) + CUDA.threadIdx().x
    if site_idx > Base.length(spiked)
        return nothing
    end
    if !spiked[site_idx]
        return nothing
    end
    synapse_idxs = siteToSyn[site_idx]
    N = Base.length(synapse_idxs)
    threads = min(N, 256)
    blocks = (N + 255) ÷ 256
    #@CUDA.cuprintln("Neuron ", site_idx, " spiked: ", N, " targets to process")
    #@CUDA.cuprintln("Length of V: ", Base.length(postVars.V))

    #There seems to be a bug (?) forbidding more than 5 arguments, workaround:
    vars = (localVars=localVars, preVars=preVars, postVars=postVars, globalVars=globalVars)
    @CUDA.cuda blocks=blocks threads=threads dynamic=true fcn(site_idx, synapse_idxs, synToTarget, vars)
    return nothing
end
