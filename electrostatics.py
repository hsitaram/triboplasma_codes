import scipy
import numpy as np
from constants import *

def collfreq(Te_in_K,mp):
    Te_in_ev=Te_in_K/evtemp
    mean_energ=1.5*Te_in_ev
    gasden=mp['pres']/kboltz/mp['Temp']
    freq=mp["collfreq_interp"](mean_energ)*gasden
    return(freq)

def bulkpot(mp,q,z):
    bpflag=mp['bulkpot']
    numden=mp['solidsvfrac']/(4/3*np.pi*mp['rp']**3)
    Vb=numden*q/(2.0*eps0)
    L=mp['cht']
    Vb=Vb*(z*L-z**2)
    return(Vb*bpflag)

def paschen(B,C,p,z):
    Vb=B*(p*z)/(C+np.log(p*z))
    return(Vb)

def imagepot(q,rp,z,mp):
    Vim=q/(2.0*np.pi*eps0)*z/(rp*(rp+2*z))
    return(Vim)

def imagebulkpot(q,rp,z,mp):
    Vimb=q/(2.0*np.pi*eps0)*z/(rp*(rp+2*z))+bulkpot(mp,q,z)
    return(Vimb)

def paschen_arr(mp,z):
    N=len(z)
    Vbr=np.zeros(N)
    for i in range(N):
        Vbr[i]=paschen(mp['B'],mp['C'],mp['pres'],z[i])
    return(Vbr)

def imagebulkpot_arr(q,rp,z,mp):
    N=len(z)
    Vimg1=np.zeros(N)
    Vimg2=np.zeros(N)
    for i in range(N):
        Vimg1[i]=imagepot(q,rp,z[i],mp)
        Vimg2[i]=imagebulkpot(q,rp,z[i],mp)
    return(Vimg1,Vimg2)

def get_acoll(vp,mp):
    A_coll=1.36*(mp['el_p']*mp['rho_part'])**(2/5)
    A_coll*=4.0*(mp['rp']**2)*(vp**(4/5))
    return(A_coll)

def contact_charge(q,vp,mp):
    A_coll=get_acoll(vp,mp)
    #print("collratio:",A_coll/(np.pi*mp['rp']**2))
    Vc=mp['delphi']
    Vb=bulkpot(mp,q,mp['delc'])
    Ve=imagepot(q,mp['rp'],mp['delc'],mp)
    #Ve=0.0
    print("voltages:",Vc,Ve,Vb)
    delq=eps0*A_coll/mp['delc']*(Vc-Ve-Vb) #Vb included in Ve
    delq=delq*mp['k_c']
    return(delq)
