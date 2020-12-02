import numpy as np

def test_import():
    from rockpool.nn.modules.nest.iaf_nest import FFIAFNest
    from rockpool.nn.modules.nest.iaf_nest import RecIAFSpkInNest 
    from rockpool.nn.modules.nest.iaf_nest import RecAEIFSpkInNest

def test_init_FFIAFNest():
    from rockpool.nn.modules.nest.iaf_nest import FFIAFNest
    
    N_in = 2
    N_rec = 3
    weights = np.random.rand(N_in, N_rec)
    
    lyr = FFIAFNest(weights=weights)

def test_evolve_FFIAFNest():
    from rockpool.nn.modules.nest.iaf_nest import FFIAFNest
    from rockpool.timeseries import TSContinuous
    
    T = 4
    N_in = 2
    N_rec = 3
    weights = np.random.rand(N_in, N_rec)
    
    lyr = FFIAFNest(weights=weights)
    
    inp = np.random.rand(T, N_in)
    ts_inp = TSContinuous.from_clocked(inp, dt=0.1, t_start=0)
    out, states, rec = lyr(ts_inp)
     
def test_evolve_RecIAFSpkInNest():
    from rockpool.nn.modules.nest.iaf_nest import RecIAFSpkInNest 
    N_in = 2
    N_rec = 3
    w_in = np.random.rand(N_in, N_rec)
    w_rec = np.random.rand(N_rec, N_rec)
    
    lyr = RecIAFSpkInNest(weights_in=w_in,
                          weights_rec=w_rec,)


def test_evolve_RecIAFSpkInNest():
    from rockpool.nn.modules.nest.iaf_nest import RecIAFSpkInNest 
    from rockpool.timeseries import TSEvent
    
    T = 100 
    N_spks = 10
    
    N_in = 2
    N_rec = 3
    w_in = np.random.rand(N_in, N_rec)
    w_rec = np.random.rand(N_rec, N_rec)
    
    lyr = RecIAFSpkInNest(weights_in=w_in,
                          weights_rec=w_rec,
                          dt=0.001)
    
    times = np.sort(np.round(np.random.rand(N_spks) * T) * lyr.dt)
    times = np.clip(times, lyr.dt, np.inf)
    channels = np.random.randint(N_in, size=(N_spks))
    
    ts_inp = TSEvent(times, channels, t_stop = (T+1) * lyr.dt)
    out, states, rec = lyr(ts_inp)
    
    
def test_evolve_RecAEIFSpkInNest():
    from rockpool.nn.modules.nest.iaf_nest import RecAEIFSpkInNest 
    N_in = 2
    N_rec = 3
    w_in = np.random.rand(N_in, N_rec)
    w_rec = np.random.rand(N_rec, N_rec)
    
    lyr = RecAEIFSpkInNest(weights_in=w_in,
                           weights_rec=w_rec,)


def test_evolve_RecAEIFSpkInNest():
    from rockpool.nn.modules.nest.iaf_nest import RecAEIFSpkInNest 
    from rockpool.timeseries import TSEvent
    
    T = 100 
    N_spks = 10
    
    N_in = 2
    N_rec = 3
    w_in = np.random.rand(N_in, N_rec)
    w_rec = np.random.rand(N_rec, N_rec)
    
    lyr = RecAEIFSpkInNest(weights_in=w_in,
                           weights_rec=w_rec,
                           dt=0.001)
    
    times = np.sort(np.round(np.random.rand(N_spks) * T) * lyr.dt)
    times = np.clip(times, lyr.dt, np.inf)
    channels = np.random.randint(N_in, size=(N_spks))
    
    ts_inp = TSEvent(times, channels, t_stop = (T+1) * lyr.dt)
    out, states, rec = lyr(ts_inp)
    
    
