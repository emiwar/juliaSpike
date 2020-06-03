import nest

pop = nest.Create("iaf_psc_exp", 2000)
syn = nest.Connect(pop, pop, {'rule': 'pairwise_bernoulli', 'p': 0.5}, {'weight': 3.0})
poisson_drive = nest.Create("poisson_generator", 1, {'rate': 1000.0})
nest.Connect(poisson_drive, pop, 'all_to_all')
spike_detector = nest.Create("spike_detector", params={'to_memory': True,
                                                       'withgid': True})
nest.Connect(pop, spike_detector)
nest.Simulate(1000.0)
print(nest.GetStatus(spike_detector, 'events'))