def modular_attractor_model1(trial_type,stim_type):
    
    import matplotlib.pyplot as plt
    import numpy as np
    n = 4;
    amp = 1.0;
    tau = .1;
    t = np.linspace(0,2,2000);
    dt = t[1]-t[0]
    inp = 0*t;
    inp[50:101] = 1;
    vL = np.true_divide([1,0,1,0],10)
    vR = np.true_divide([0,1,0,1],10)
    Vin = np.zeros([2,4])
    Vin[0,:] = vL
    Vin[1,:] = vR
    r = np.ones((np.size(t),n,2))*.7
        
    w = np.array([[.5936,-.4947],[-.4947,.5936]])
    cc = 0.3
    W = np.zeros((4,4))
    W[0:2,0:2] = w
    W[2:4,2:4] = w
    W[0,2] = cc;W[2,0]=cc;W[1,3]=cc;W[3,1]=cc
    I = [.5,.5,.5,.5]
    simtime = [200,1000]
    stim_index = np.zeros((4,), dtype=np.object)
    stim_index[0] = []
    stim_index[1] = np.array([0,1])
    stim_index[2] = np.array([2,3])
    stim_index[3] = np.array([0,1,2,3])
    
    
    r=np.ones((np.size(t),n))*.65   
    for i in range(1,np.size(t)):
        a=r[i-1,:]
        denom = amp+amp*np.exp(-a/.3+.5/.3)
        ar = np.true_divide(1.4*amp,denom)-.2224
        a=a+dt/tau*(-a+np.dot(ar,W)+inp[i]*Vin[trial_type,:]+I+np.random.randn(1,n)*.1)
        if i>simtime[0] and i<simtime[1]:
            a[0][stim_index[stim_type]]=0
        
        r[i,:]=a
        
    plt.plot(r[:,1])
    plt.plot(r[:,0])
    
def modular_integrator_model(stim_type):
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.linspace(0,2.2,2201)
    dt = t[1]-t[0]
    tau = .01
    noise = 0
    n = 8
    
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
                 
    vin = np.array([1,-1,0,0,1,-1,0,0])
    ton = np.array([1,1,0,0,1,1,0,0])*2
    pv = np.zeros([1,n])
    stim_index = np.zeros((4,), dtype=np.object)
    stim_index[0] = []
    stim_index[1] = [0,1,2,3]
    stim_index[2] = [4,5,6,7]
    stim_index[3] = [0,1,2,3,4,5,6,7]
    
    pv[0][stim_index[stim_type]]=1
    r = np.zeros([np.size(t),n])
    p = 0*t
    p[400:1201] = -10
    inn = 0*t;
    inn[100:np.size(t)] = .01
    
    
    for i in range(1,np.size(t)):
        a=r[i-1,:]
        a = a + dt/tau*(-a + np.dot(a,w) + inn[i]*vin + p[i]*pv*10 + ton*2 + np.random.randn(1,n)*noise)
        a[a<0] = 0
        r[i,:]=a
        
        plt.plot(r[:,0])
        
    
def modular_rnn1(stim_type):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal
    
    Q = 1
    n_loops = 10
    
    N = 400
    p = 0.1
    g = 1.5
    alpha = 1.0
    nsecs = 2000
    dt = 1.0 
    tau = 200.0
    learn_every = 2
    
    scale = 1.0/np.sqrt(p*N)
    M_mask = np.random.rand(N,N)
    M_mask[M_mask > p] = 1
    M_mask[M_mask <= p] = 0
    M_mask = 1 - M_mask
    M = np.random.randn(N,N)
    M = np.multiply(M,M_mask)
    
    pop = np.zeros((2,), dtype=np.object)    
    pop[0] = np.arange(0,N/2,1)
    pop[1] = np.arange(N/2+1,N,1)
    
    
    nRec2Out = N;
    wo = np.zeros([nRec2Out,2]);
    w_in=np.random.randn(N,1);
    dw = np.zeros([nRec2Out,1]);
    wf = 2.0*(np.random.rand(N,1)-0.5);
    
    
    simtime = np.arange(0,nsecs - dt,dt)
    simtime_len = np.size(simtime);
    simtime2 = np.arange(nsecs,2*nsecs - dt,dt)


    input1 = 0*simtime
    input1[100:200] = 2    
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
    
    
    zt = np.zeros([1,simtime_len])    
    zpt = np.zeros([1,simtime_len])
    z0 = 0.5*np.random.randn(N,1)*0
    x0 = 0.5*np.random.randn(N,1)*0
    
    M0 = M
    noise = 0
    
    u = np.zeros([N,2])
    u[pop[0],0] = 1
    u[pop[1],1] = 1
    
    rec = np.array([[1,2,2],[2,1,1]]).T - 1 
    readout = np.array([[1,1,1],[2,2,2]]).T - 1
    
    R = np.zeros([N,np.size(simtime),n_loops,2,3])
    Z = np.zeros([np.size(simtime),n_loops,2,3])
    for trn in range(1,n_loops):
        for LorR in range(0,2):
            for stim_on in range(2,-1,-1):
                ft = FT[LorR][stim_on,:]
                x = x0*1.0
                r = np.tanh(x)
                z = np.dot(wo[:,LorR].T,r)
                
                ti = -1
                P = (1.0/alpha)*np.eye(N)
                clamp = np.zeros([N,1])
                
                for j in range(int(1),int(1999)):
                    
                    ti = ti + 1
                        
                    x = (1.0-dt/tau)*x + np.dot(M,r*dt/tau) + dt/tau*input1[ti]*w_in + \
                    np.random.randn(N,1)*noise                
                
                    if ti>200 and ti<1000:
                        if stim_on == 1:                
                            x[0:N/2] = 0
                        
                        if stim_on == 2:
                            x[N/2:N] = 0
               
                    r = np.tanh(x) 
                    z = np.dot(wo[:,LorR].T,r)
               
                    if trn>0 and trn<n_loops-1: 
                        if np.mod(ti,learn_every) == 0:
                            print[trn,stim_on,LorR,j]
                            rr = r*0
                            rr[pop[readout[stim_on,LorR]]] = r[pop[readout[stim_on,LorR]]]
                            k = np.dot(P,rr)
                            rPr = np.dot(rr.T,k)
                            c = np.true_divide(1.0,(1.0 + rPr))
                            P = P - np.dot(k,np.dot(c,k.T))
                            
                            e = z - ft[ti]
                            
                            dw = -e*k*c
                            wo[:,LorR] = wo[:,LorR] + dw[:,0]
                            M = M + np.outer(u[:,rec[stim_on,LorR]],dw[:,0].T)
                      
                    Z[ti,trn,LorR,stim_on] = z
                
                        
    
    
    
    
        
    