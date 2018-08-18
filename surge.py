#!/usr/bin/env python3
import sys,os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import ode, solve_ivp
import tools

mpl,fntsize,labsize = tools.set_mpl(mpl)
##########################################################################

def surge_mm(a=None,b=None,mun=None,dc=None,ch=None,pw0=None,pwinfty=None,pwr=None,phi0=None):
    p = Parameters()
    p.reconcile_inputs(a,b,mun,dc,ch,pw0,pwinfty,pwr,phi0)
    s2yr = p.s2y

    t0 = 0.
    tf = 100.*86400.*365.
    numsteps = 2**10
    i_numsteps = int(numsteps+0.1)

    dt = (tf - t0)/float(numsteps)
    t_array = np.linspace(t0,tf+dt,i_numsteps)
    t_array[-1] = tf
    t_array_yr = t_array*s2yr

    pwbalance = p.pw0  ### already normalized 
    alphabalance = p.alpha0
    phibalance = p.phi0
    mubalance = p.mun

    p.ubnorm_term = (alphabalance - mubalance*(1.-pwbalance))**p.n
    p.ubref = 2.*p.A*(p.rhoi*p.g)**p.n * p.w**(p.n + 1) / (p.n + 1)
    p.ubbalance = p.ubref * p.ubnorm_term
    p.ubc = p.ubbalance

    thetabalance = p.dc / p.ubbalance
    p.thetac = p.dc / p.ubc

    alpha = alphabalance

    ubinit = 1.1*p.ubc
    thetainit = thetabalance
    thickinit = p.h
    p_over = p.rhoi*p.g*thickinit
    pwinit = pwbalance * p_over #np.min([1., 0.05 + pwbalance])
    phiinit = phibalance
    alphainit = alphabalance

    y0 = [ubinit,thetainit,pwinit,phiinit,thickinit,alphainit]

    otv = solve_ivp(lambda t,y: surge_solver(t,y,p),[t0,tf],y0,method='Radau',
                         dense_output=True,max_step=dt)

    fig = plt.figure(figsize=(12,8))
    plt.plot(otv.t*s2yr, otv.y[0]/p.ubc,'.',linestyle='-',label='$u_b$')
    plt.plot(otv.t*s2yr, otv.y[1]/thetabalance,label=r'$\theta$')
    plt.plot(otv.t*s2yr, otv.y[2]/p_over,label=r'$p_w$')
    plt.plot(otv.t*s2yr, otv.y[3],label=r'$\phi$')
    plt.plot(otv.t*s2yr, otv.y[4]/p.h,linewidth=4,label=r'$h$')
    plt.plot(otv.t*s2yr, otv.y[5]/alphabalance,label=r'$\alpha$')
    plt.plot(otv.t*s2yr, 1.+otv.t*0.,'--',color='0.6')
    plt.legend()
    plt.ylim([0,10])
    plt.xlabel('Time (years)')
    plt.savefig('figs/ub_v_t/ub_vs_time_01.pdf',bbox_inches='tight')

def surge_solver(t,y,p):
    eps = p.machine_eps

    ub = np.max([eps, y[0]])
    theta = np.max([eps, y[1]])
    pw_Pa = np.max([eps, y[2]])
    phi = np.max([eps, y[3]])
    h = np.max([eps, y[4]])
    alpha = np.max([eps, y[5]])
    p_over = p.rhoi*p.g*h

    ### nondimensional parameters
    pw = np.min([pw_Pa/p_over, 1.-eps])

    N = 1. - pw
    Lambda = p.epsilon_e * (1. - phi)**2 / p.epsilon_p  ### dimensionless
    beta = p.epsilon_p * Lambda / N   

    ### state
    tdotarg = ub * theta / p.dc
    thetadot = -tdotarg * np.log(tdotarg) 

    ### water pressure
    pwdot_st = p.ch * (p.pwinfty + p.pwr - 2.*pw)
    pwdot_dy = thetadot * N / (theta * Lambda) 
    pwdot = p_over * (pwdot_st + pwdot_dy)

    ### rate and state friction coef
    mu = p.mun + p.a * np.log(ub/p.ubc) + p.b * np.log(theta*p.ubc/p.dc)

    ### porosity
    phidot_e = pwdot * beta / p_over
    phidot_p = -p.epsilon_p * thetadot / theta
    phidot = phidot_e + phidot_p

    ### glacier geometry
    hdot = p.zeta * alpha * (p.ubbalance - ub) 
    alphadot = hdot * alpha / h

    ### acceleration 
    ubdot = p.n * ub * ((alphadot - mu*(pw*hdot/h - pwdot/p_over) - 
                    N*p.b*thetadot / theta) / (alpha + (p.a*p.n - mu)*N))

    return [ubdot, thetadot, pwdot, phidot, hdot, alphadot]

class Parameters():
    def __init__(self):
        ### material properties of ice
        self.n = 3.
        self.A = 2.4e-24
        self.rhoi = 900.
        self.g = 9.81

        ### glacier geometry
        self.h = 300.
        self.w = 500.
        self.alpha0 = 5.e-2

        ### rate and state parameters
        self.mun = 0.3
        self.dc = 1.
        self.ab = 0.9
        self.b = 0.15
        self.a = self.b*self.ab

        ### ratio of depth averaged velocity to surface velocity (0.8 <= zeta <= 1)
        self.zeta = 1.

        ### till properties 
        self.epsrat = 60.
        self.epsilon_p = 1.e-4
        self.epsilon_e = self.epsrat*self.epsilon_p 
        self.ch = 1.e-10 #self.kappah/self.hs**2

        self.phi0 = 0.1
        self.pw0 = 0.9  ### pw0/pi

        self.pwinfty = self.pw0 ### pwinfty/pi
        self.pwr = self.pw0     ### pwr/pi

        ### misc 
        self.s2y = 365.*86400.
        self.y2s = 1./self.s2y
        self.machine_eps = np.finfo(np.float64).eps

    def reconcile_inputs(self,a,b,mun,dc,ch,pw0,pwinfty,pwr,phi0):
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        if mun is not None:
            self.mun = mun
        if dc is not None:
            self.d_c = dc
        if ch is not None:
            self.ch = ch
        if pw0 is not None:
            self.pw0 = pw0
        if pwinfty is not None:
            self.pwinfty = pwinfty
        if pwr is not None:
            self.pwr = pwr
        if phi0 is not None:
            self.phi0 = phi0

if __name__=='__main__':
    main()
