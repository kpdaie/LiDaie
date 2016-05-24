def modular_attractor_model(trial_type,stim_type):
    import matplotlib.pyplot as plt
    import numpy as np

    n = 4;#number of neurons
    amp = 1.0;#amplitude of non-linearity
    tau = .1;#synaptic time constant
    t = np.linspace(0,2,2000);#time points
    dt = t[1]-t[0]#time step
    inp = 0*t;
    inp[50:101] = 1;#external input
    vL = np.true_divide([1,0,1,0],10)#left spatial vector
    vR = np.true_divide([0,1,0,1],10)#right spatial vecotr
    Vin = np.zeros([2,4])
    Vin[0,:] = vL
    Vin[1,:] = vR
    r = np.ones((np.size(t),n,2))*.7
        
    w = np.array([[.5936,-.4947],[-.4947,.5936]])#matrix of intra-modular connections
    cc = 0.3#strength of inter-modular connections
    W = np.zeros((4,4))
    W[0:2,0:2] = w
    W[2:4,2:4] = w
    W[0,2] = cc;W[2,0]=cc;W[1,3]=cc;W[3,1]=cc#full connectivity matrix
    I = [.5,.5,.5,.5]#strength of tonic inputs to each neuron
    simtime = [200,1000]#start and stop time points for the perturbation
    stim_index = np.zeros((4,), dtype=np.object)#this is a list of the different spatial profiles for the perturbation
    stim_index[0] = []#no perturbation (stim_type = 0)
    stim_index[1] = np.array([0,1])#perturb left module
    stim_index[2] = np.array([2,3])#perturb right module
    stim_index[3] = np.array([0,1,2,3])#perturb both modules
    
    
    r=np.ones((np.size(t),n))*.65#initial firing rate   
    for i in range(1,np.size(t)):#solve the dynamics using Euler's method
        a=r[i-1,:]
        denom = amp+amp*np.exp(-a/.3+.5/.3)#non-linearity 
        ar = np.true_divide(1.4*amp,denom)-.2224#non-linearity 
        a=a+dt/tau*(-a+np.dot(ar,W)+inp[i]*Vin[trial_type,:]+I+np.random.randn(1,n)*.1)
        if i>simtime[0] and i<simtime[1]:#apply perturbation
            a[0][stim_index[stim_type]]=0
        
        r[i,:]=a
        
    plt.plot(t,r[:,1])
    plt.plot(t,r[:,0])
    
def modular_integrator_model(stim_type):
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.linspace(0,2.2,2201)#time points
    dt = t[1]-t[0]#time step
    tau = .01#synaptic time constant
    noise = 0#noise amplitude
    n = 8#number of neurons
    
    #defining the connectivity matrix
    rec1 = .5
    rec2 = -.5
    inh = -6
    w = np.matrix([[0,-1,1,1,0,0,0,0],
                   [-1,0,-1,-1,0,0,0,0],
                   [0,0,0,0,rec1,rec2,0,0],
                   [0,0,0,0,0,0,inh,0],
                   [0,0,0,0,0,-1,1,1],
                   [0,0,0,0,-1,0,-1,-1],
                   [rec1,rec2,0,0,0,0,0,0],
                   [0,0,inh,0,0,0,0,0]])
                 
    vin = np.array([1,-1,0,0,1,-1,0,0])#spatial vector of external inputs
    ton = np.array([1,1,0,0,1,1,0,0])*2#spatial vector of tonic inputs
    pv = np.zeros([1,n])
    #spatial vectors of perturbed neurons
    stim_index = np.zeros((4,), dtype=np.object)
    stim_index[0] = []
    stim_index[1] = [0,1,2,3]
    stim_index[2] = [4,5,6,7]
    stim_index[3] = [0,1,2,3,4,5,6,7]
    
    pv[0][stim_index[stim_type]]=1
    r = np.zeros([np.size(t),n])#initial firing rates
    p = 0*t
    p[400:1201] = -10#perturbation time series
    inn = 0*t;
    inn[100:np.size(t)] = .01#external input time series
    
    
    for i in range(1,np.size(t)):#solve dynamics using Euler's method
        a=r[i-1,:]
        a = a + dt/tau*(-a + np.dot(a,w) + inn[i]*vin + p[i]*pv*10 + ton*2 + np.random.randn(1,n)*noise)
        a[a<0] = 0#make sure activity doesn't go negative
        r[i,:]=a
        
    plt.xlim(.05, 2.05)
    plt.plot(t-.05,r[:,0])
       
    
def modular_rnn_model(n_loops):    
    import matplotlib.pyplot as plt
    import numpy as np
    
    N = 400#number of neurons
    p = 0.1#connection probability
    g = 1.5#scalar strength of connections (>=1.5 gives chaotic behavior)
    alpha = 1.0#learning rate
    nsecs = 2000#number of milliseconds
    dt = 1.0#time step in milliseconds 
    tau = 200.0#synaptic time constant in milliseconds
    learn_every = 2#train new weights every (learn_every) time steps
    
    #define the initial random chaotic connectivity matrix
    scale = 1.0/np.sqrt(p*N)
    M_mask = np.random.rand(N,N)
    M_mask[M_mask > p] = 1
    M_mask[M_mask <= p] = 0
    M_mask = 1 - M_mask
    M = np.random.randn(N,N)
    M = np.multiply(M,M_mask)*g*scale
    
    #break the neurons into two modules
    pop = np.zeros((2,), dtype=np.object)    
    pop[0] = np.arange(0,N/2,1)
    pop[1] = np.arange(N/2+1,N,1)
    
    
    wo = np.zeros([N,2]);#readout vectors (one for each module)
    w_in=np.random.randn(N,1);#spatial vector of external inputs
    dw = np.zeros([N,1]);#change in readout vector
#    wf = 2.0*(np.random.rand(N,1)-0.5);
    
 #time vector   
    simtime = np.arange(0,nsecs - dt,dt)
    simtime_len = np.size(simtime);
    simtime2 = np.arange(nsecs,2*nsecs - dt,dt)
    
    #external input time series
    input1 = 0*simtime
    input1[100:200] = 2    
    #code to generate ramping output target functions 
    f = np.cumsum(input1)*dt/tau/100.0*1.3
    f = np.cumsum(f)*dt/tau
    
    inddd = np.arange(200,1000,1)    
    f2 = f*1.0
    f2[inddd] = 0
    ft = np.zeros([3,np.size(f)])
    ft2 = 0*ft
    FT = np.zeros((2,), dtype=np.object)
    ft[0,:] = f
    ft[1,:] = f2
    ft[2,:] = f
    ft2[0,:] = f
    ft2[1,:] = f
    ft2[2,:] = f2
    FT[0] = ft    
    FT[1] = ft2
    
    
    #zt = np.zeros([1,simtime_len])    
    #zpt = np.zeros([1,simtime_len])
    #z0 = 0.5*np.random.randn(N,1)*0
    x0 = 0.5*np.random.randn(N,1)*0#initial activity
    
    M0 = M
    noise = 0
    
    #readin vectors 
    u = np.zeros([N,2])
    u[pop[0],0] = 1
    u[pop[1],1] = 1
    
    #indices used for controlling which connections are trained during a particular condition (more explanation below)
    rec = np.array([[1,2,2],[2,1,1]]).T - 1 
    readout = np.array([[1,1,1],[2,2,2]]).T - 1
    
    #initialize rate and readout
    R = np.zeros([N,np.size(simtime),n_loops,2,3])
    Z = np.zeros([np.size(simtime),n_loops,2,3])
    for trn in range(1,n_loops):#training iteration loop
        for LorR in range(0,2):#train either left or right module's readout
            for stim_on in range(2,-1,-1):#stim_on 2 (silence right); stim_on = 1 (silence left);stim_on = 0 (no perturbation)
                ft = FT[LorR][stim_on,:]
                x = x0*1.0
                r = np.tanh(x)
                z = np.dot(wo[:,LorR].T,r)
                
                ti = -1
                P = (1.0/alpha)*np.eye(N)
                clamp = np.zeros([N,1])
                
                for j in range(int(1),int(1999)):#solve dynamics using Euler's method
                    ti = ti + 1
					
                    x = (1.0-dt/tau)*x + np.dot(M,r*dt/tau) + dt/tau*input1[ti]*w_in + \
                    np.random.randn(N,1)*noise                
                    
                    if ti>200 and ti<1000:#apply perturbation
                        if stim_on == 1:                
                            x[0:N/2] = 0
                        if stim_on == 2:
                            x[N/2:N] = 0
					
                    r = np.tanh(x) 
                    z = np.dot(wo[:,LorR].T,r)
		   
                    if trn>0 and trn<n_loops-1:#training section 
                        if np.mod(ti,learn_every) == 0:
                            rr = r*0
                            #calculating the change in readout weights according to 
                            #eqs 4 and 5 in Sussillo, 2009. 
                            #rr is a vector of rates for a subset of neurons which determines
                            #the population of postsynaptic connections to be trained
                            rr[pop[readout[stim_on,LorR]]] = r[pop[readout[stim_on,LorR]]]
                            k = np.dot(P,rr)
                            rPr = np.dot(rr.T,k)
                            c = np.true_divide(1.0,(1.0 + rPr))
                            P = P - np.dot(k,np.dot(c,k.T))
                            e = z - ft[ti]
                            dw = -e*k*c
                            wo[:,LorR] = wo[:,LorR] + dw[:,0]
                            #here the outer product of the change of readout weights 
                            #is taken with the readin weight vector to determine the
                            #change in recurrent connections.
                            #the readout weights determine which postsynaptic connections are trained
                            #the readin weights determine which presynaptic connections are trained
                            #by separating these we are able to train inter- and intra- modular connections 
                            #separately as described in the Methods. 
                            M = M + np.outer(u[:,rec[stim_on,LorR]],dw[:,0].T)
                    Z[ti,trn,LorR,stim_on] = z
    plt.xlim(0, 1.9)
    plt.plot(simtime/1000,Z[:,n_loops-1,0,[0,1,2]])