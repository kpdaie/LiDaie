def Simple_integrator_model(frac_silence)
    n = 200
    r0 = 5
    rm = 100
    v = np.random.rand(n)
    pert = np.random.permutation(n)
    num_pert = int(np.round(n*frac_silence))
    pert = pert[1:num_pert]
    
    w = np.outer(v.T,v)
    eigval, eigvec = np.linalg.eig(w)
    w = w/max(eigval)*.99
    w = np.real(w)

    dt = .001
    tau = .1
    t = np.linspace(0,2,2001)
    inp = 0*t
    inp[100:1900] = 1
    r = np.zeros([np.size(t),n])
    
    q = 1
    for i in range(1,np.size(t)):
        a=r[i-1,:]
        a = a + dt/tau*(-a + np.dot(a,w) + inp[i]*v*2)
        if i>200 and i<1000:
            a[pert] = 0
        a[a<0] = 0
        r[i,:]=a
        






def Integrator_with_corrective_feedback(trial_type,stim_type)
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.linspace(0,2,200001)
    dt = t[1]-t[0]
    taue = 0.02
    taui = 0.01
    
    tee=.1
    tie=.025
    tii=.01
    tei=.01

    inp=np.cumsum(np.exp(-np.square((t-.1))/.005))/10000
    inp2=0*t
    inp2[20001:100001]=4000
    
    Jee = 150.0
    Jii = 300.0
    Jie = 150.0
    Jei = 300.0
    
    Jeo = 1500.0
    Jio = 0.0
    stim_amp = np.array([0,1])
    trial_amp = np.array([1,-1])
    
    q = 1
    re = np.array([0])
    ri = np.array([0])
    see = 0
    sei = 0
    sie = 0
    sii = 0
    
    RE = np.zeros([np.size(t)])
    RI = 0*RE
    SEE = 0*RE
    SEI = 0*RE
    for i in range(1,np.size(t)):
        Dre = -re + Jee*see - Jei*sei + Jeo*inp[i]*trial_amp[trial_type]
        Dri = -ri - Jii*sii + Jie*sie + Jio*inp[i]*trial_amp[trial_type] + inp2[i]*stim_amp[stim_type]
        Dee = -see + re
        Dei = -sei + ri
        Die = -sie + re
        Dii = -sii + ri
        
        re = re + dt/taue * Dre
        ri = ri + dt/taui * Dri
        
        re[re<0] = 0
        ri[ri<0] = 0
        
        see = see + dt/tee * Dee
        sii = sii + dt/tii * Dii
        sie = sie + dt/tie * Die
        sei = sei + dt/tei * Dei
        
        RE[i] = re
        RI[i] = ri
        SEE[i] = see
        SEI[i] = sei

def FORCE_rnn(n_loops,n_trials):
    import matplotlib.pyplot as plt
    import numpy as np
    
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
    M = np.multiply(M,M_mask)*g*scale
    
    pop = np.zeros((2,), dtype=np.object)    
    pop[0] = np.arange(0,N,1)
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
    
    
    for trn in range(0,n_loops+n_trials):
        for LorR in [0]:
            for stim_on in [0]:
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
                    
                    if trn >= n_loops-1:                    
                        if ti>200 and ti<1000:
                            if stim_on == 1: 
                                aaa = 1
                                perm = np.random.permutation(N)
                                x[perm[0:N/2]] = 0
                        
					
                    r = np.tanh(x) 
                    z = np.dot(wo[:,LorR].T,r)
		   
                    if trn>0 and trn<n_loops-1: 
                        if np.mod(ti,learn_every) == 0:
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
    
    Z = np.zeros([np.size(simtime),n_trials])
    for trl in range(0,n_trials - 1):      
        trl
        ti = 0;
        perm = np.random.permutation(N)
        x = x0
        r = np.tanh(x)                                                   
        for j in range(int(1),int(1999)):
                    ti = ti + 1
					
                    x = (1.0-dt/tau)*x + np.dot(M,r*dt/tau) + dt/tau*input1[ti]*w_in + \
                    np.random.randn(N,1)*noise                
                    
                    if ti>200 and ti<1000:
                        if trl > 0:                                                         
                            x[perm[0:N/2]] = 0

                        
					
                    r = np.tanh(x) 
                    z = np.dot(wo[:,0].T,r)
                    Z[ti,trl] = z                    
    plt.plot(simtime/1000,Z)