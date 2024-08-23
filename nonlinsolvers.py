import scipy
import numpy as np
from scipy.optimize import fsolve
from electrostatics import *

def arrh_rate(A,alpha,Ta_in_K,T_in_K):
    rateconst=A* (T_in_K**alpha) * np.exp(-Ta_in_K/T_in_K)
    return(rateconst)

def solve_tangent(soln,mp):
    q=soln[0]
    z=soln[1]
    paschen_der=mp['B']*mp['pres']*(mp['C']+np.log(mp['pres']*z)-1)/((mp['C']+np.log(mp['pres']*z))**2.0)
    image_der=q/(2.0*np.pi*eps0)/((mp['rp']+2*z)**2)

    numden=mp['solidsvfrac']/(4/3*np.pi*mp['rp']**3)
    L=mp['cht']
    Vbulk_der=numden*q/(2.0*eps0)*(L-2.0*z)
    
    f1=paschen_der-image_der-mp['bulkpot']*Vbulk_der
    f2=paschen(mp['B'],mp['C'],mp['pres'],z)-imagepot(q,mp['rp'],z,mp)-mp['bulkpot']*bulkpot(mp,q,z)
    return([f1,f2])

def tangent_solve_bisection(qg1,qg2,mp,N=1000000,itmax=20):
    z = np.linspace(0.37*mp['dc_SI'],10000.0*mp['dc_SI'],N)
    qbs1=qg1 #no intersection (bs is bisection search)
    qbs2=qg2 #intesects
    loc=z[0]
    for it in range(itmax):
        Vbr=paschen_arr(mp,z)
        qmid=0.5*(qbs1+qbs2)
        (pot_img,pot_total)=imagebulkpot_arr(qmid,mp['rp'],z,mp)
        idx = np.argwhere(np.diff(np.sign(Vbr - pot_total))).flatten()
        if(len(idx)==0):
            qbs1=qmid
        else:
            qbs2=qmid
            loc=z[idx][0]
            if(len(idx)>1):
                loc=0.5*(z[idx][0]+z[idx][1])
        print(it,qmid,loc)
    return(0.5*(qbs1+qbs2),loc)

def intersect_solve_graphical(q,mp,N=1000000):
    z = np.linspace(0.37*mp['dc_SI'],1000.0*mp['dc_SI'],N)
    Vbr=paschen_arr(mp,z)
    (pot_img,pot_total)=imagebulkpot_arr(q,mp['rp'],z,mp)
    idx = np.argwhere(np.diff(np.sign(Vbr - pot_total))).flatten()
    intersects=True
    if(len(idx)==0):
        intersects=False
    if(intersects):
        return(intersects,z[idx])
    else:
        return(intersects,np.array([-1.0]))


def solve_intersect(soln,q,mp):
    z=soln[0]
    f=paschen(mp['B'],mp['C'],mp['pres'],z)-imagebulkpot(q,mp['rp'],z,mp)
    return([f])

def solve_intersect_q(soln,z,mp):
    q=soln[0]
    f=paschen(mp['B'],mp['C'],mp['pres'],z)-imagebulkpot(q,mp['rp'],z,mp)
    return([f])

def solve_Te_electronflux(soln,z,mp):
    Te=soln[0]
    vol=np.pi*(mp['rp']**2)*z
    A=np.pi*(mp['rp']**2)
    rate_ionize=arrh_rate(mp['Aiz'],mp['alphaiz'],mp['Ta_iz'],Te)
    cbar=np.sqrt(8.0*kboltz*Te/np.pi/melec)
    f=rate_ionize*mp['NG']*vol-(cbar/2.0)*A
    return([f])

def solve_Te_bohmflux(soln,z,mp):
    Te=soln[0]
    vol=np.pi*(mp['rp']**2)*z
    A=np.pi*(mp['rp']**2)
    rate_ionize=arrh_rate(mp['Aiz'],mp['alphaiz'],mp['Ta_iz'],Te)
    uB=np.sqrt(kboltz*Te/mp['m_ion']) #ion bohm speed
    f=rate_ionize*mp['NG']*vol-uB*A*0.5
    return([f])

def solve_Te_ambipolar(soln,z,mp):
    Te=soln[0]
    Da=mp['D_i']*(1.0+Te/mp['Temp'])
    beta2=np.pi**2/z**2
    rate_ionize=arrh_rate(mp['Aiz'],mp['alphaiz'],mp['Ta_iz'],Te)
    #solve
    #Kiz ng/Da = pi^2/l^2
    f=rate_ionize*mp['NG']-beta2*Da
    return([f])

def solve_Te(soln,z,mp):
    return(solve_Te_ambipolar(soln,z,mp))
    #return(solve_Te_bohmflux(soln,z,mp))
    #return(solve_Te_electronflux(soln,z,mp))
