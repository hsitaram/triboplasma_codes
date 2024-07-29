import scipy
import numpy as np

def gaussdist(x,mu,sig):
    f=1/np.sqrt(2.0*np.pi)/sig*np.exp(-(x-mu)**2/2/sig**2)
    return(f)

def maxwellspeeddist(x,T):
    f=np.sqrt(2.0/np.pi)*(1.0/(T**1.5))* (x**2) * np.exp(x**2/2.0/T)
    return(f)

def collidingdist(x,T):
    #f=(1.0/(2.0*np.pi*T))**(1.5) * x * np.exp(-x**2/(2.0*T)) * (2.0*np.pi*T)
    #int_f = 0.25*np.sqrt(8.0*T/np.pi)
    #return(f/int_f)
    fc=x * np.exp(-x**2/(2.0*T)) / T
    return(fc)


def speeddist(v,Tg):
    f=4.0/np.sqrt(np.pi)
    f=f*(2.0*Tg)**1.5
    f=f*(v**2)*np.exp(v**2/2.0/Tg)
    return(f)
