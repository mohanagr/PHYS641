import numpy as np

def get_deriv(func,pars,dpar,width):
    model=func(pars,width)
    npar=len(pars)
    print(npar, "len pars")
    derivs=[None]*npar
    for i in range(npar):
        pp=pars.copy()
        pp[i]=pars[i]+dpar[i]
        m_plus=func(pp,width)
        pp[i]=pars[i]-dpar[i]
        m_minus=func(pp,width)
        derivs[i]=(m_plus-m_minus)/(2*dpar[i])
    return model,derivs

def gauss2d(pars,width):
    # vec = 
    vec=np.asarray(np.arange(-width,width),dtype='float')
    #could have fftfreq convention too. doesn't matter as long as what's predicted matches what's expected.
    amp=pars[0]
    dx=pars[1]
    dy=pars[2]
    sigx=pars[3]
    sigy=pars[4]

    xvec=vec-dx
    yvec=vec-dy
    xmat=np.outer(xvec,np.ones(len(xvec)))
    ymat=np.outer(np.ones(len(yvec)),yvec)
    rmat=xmat**2/sigx**2 +ymat**2/sigy**2
    model=np.exp(-0.5*rmat)*amp

    return model

def pad_map(map):
    map=np.hstack([map,np.fliplr(map)])
    map=np.vstack([map,np.flipud(map)])
    return map

def get_model_derivs_flat(func,pars,dpar,width):
    model,derivs=get_deriv(func,pars,dpar,width)
    model=np.ravel(model)
    npar=len(pars)
    derivs_out=np.empty([len(model),len(pars)])
    for i in range(npar):
        derivs_out[:,i]=np.ravel(derivs[i])
    return model,derivs_out

def fit_lm(func, m, y, *args, N=None, lmbda=1, niter=5000, rtol=1.e-3):
    #fun: model function whose derivative will be taken
    #m: starting guess of pars
    #y: data that model is being fit to
    print("init pars are:", m)

    def update_lambda(lamda, success):
        if success:
            lamda = lamda / 1.5
            if lamda < 0.5:
                lamda = 0
        else:
            if lamda == 0:
                lamda = 1
            else:
                lamda = lamda * 2
        return lamda

    lm = lmbda  # start from a higher lambda if initial Newton steps are crap and errors are thrown

    I = np.eye(len(m))
    if (N is None):
        print("yep")
        N = np.eye(len(y))

    model, derivs = get_model_derivs_flat(func,m,*args)
    r = y - model
    Ninv = np.linalg.inv(N)
    chisq = r.T @ Ninv @ r

    for i in range(niter):

        lhs = (derivs.T @ Ninv @ derivs + lm * I)  # first step is always Newton's
        rhs = derivs.T @ Ninv @ r
        dm = np.linalg.inv(lhs) @ rhs
        m_trial = m + dm
        print('on iteration ', i, ' chisq is ', chisq, ' taking step ', m_trial, 'with lambda ', lm)
        try:
            model, derivs = get_model_derivs_flat(func,m_trial,*args)
        except Exception as e:
            print("bad params ")
            lm = update_lambda(lm, False)
            continue
        r = y - model
        chisqnew = r.T @ Ninv @ r

        if (chisqnew < chisq):

            # accept the new step
            m = m_trial
            lm = update_lambda(lm, True)
            print("step accepted. new m is", m)
            relerr = np.abs((chisqnew - chisq) / chisq)
            if (relerr < rtol and lm == 0):
                # if lm=0, we're in Newton's domain, and fairly close to actual minima
                # Even if chain coverges before lm=0, let the temperature decrease and lm reach 0 before exiting
                param_cov = np.linalg.inv(derivs.T @ Ninv @ derivs)
                print("CHAIN CONVERGED")
                break
            chisq = chisqnew

        else:
            # stay at the same point and try a more Gradient descent-ish step next
            lm = update_lambda(lm, False)
            # if rtol is too small, improvement in chisq may never reach that level. better to exit and consider
            # it converged
            if (lm > 1e8):
                param_cov = np.linalg.inv(derivs.T @ Ninv @ derivs)
                print("CHAIN STUCK. IS RTOL TOO SMALL? TERMINATING")
                break

            print("step rejected. old m is", m)
    return m, param_cov

def newton(pars,data,fun,width,dpar,niter=10):
    for i in range(niter):
        model,derivs=get_model_derivs_flat(fun,pars,dpar,width)
        resid=data-model
        lhs=derivs.T@derivs
        rhs=derivs.T@resid
        shift=np.linalg.inv(lhs)@rhs
        print('parameter shifts are ',shift)
        pars=pars+shift
    return pars

def get_kernel(map_shape, sig_smooth=100):
    xx=np.fft.fftfreq(map_shape[0])*map_shape[0]
    yy=np.fft.fftfreq(map_shape[1])*map_shape[1]
    X,Y = np.meshgrid(yy,xx)
    kernel = np.exp(-0.5*(X**2+Y**2)/sig_smooth**2)
    return kernel/kernel.sum()


def estimate_ps(mymap, sig_smooth=100):
    "Smooth the PS by 100 k modes"
    
    padmap = pad_map(mymap)
    mapft = np.fft.fft2(padmap)
    kernel = get_kernel(padmap.shape,sig_smooth=sig_smooth)
    ps = np.real(mapft*np.conj(mapft))
    smooth_ps = np.real(np.fft.ifft2(np.fft.fft2(ps)*np.fft.fft2(kernel)))
    return smooth_ps

def apply_Ninv(mymap, ps):
    padmap = pad_map(mymap)
    mapft = np.fft.fft2(padmap)
    filt_map = padmap.shape[0]*padmap.shape[1]*np.real(np.fft.ifft2(mapft/ps))
    return filt_map[:mymap.shape[0],:mymap.shape[1]].copy()



    
