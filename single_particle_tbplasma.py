import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from model_parameters import *
from electrostatics import *
from nonlinsolvers import *
from distfuncs import *

#main
font={'family':'Helvetica', 'size':'14'}
mpl.rc('font',**font)
mpl.rc('xtick',labelsize=14)
mpl.rc('ytick',labelsize=14)

mp=setmodelparams()
mp['bulkpot']=0
vp=20.0

(qt,tloc)=tangent_solve_bisection(0.0,60e-9,mp,N=100000)
print("qt,tloc",qt,tloc)

N=100000
max_z_log10=np.log10(500.0*mp['dc_SI'])
min_z_log10=np.log10(0.37*mp['dc_SI'])
zp=np.logspace(min_z_log10,max_z_log10,N)
Vbr=paschen_arr(mp,zp)
Vimg_tangent,Vtot_tangent=imagebulkpot_arr(qt,mp['rp'],zp,mp)

delq=contact_charge(qt,vp,mp)
print("delq:",delq)
Vimg_postcoll,Vtot_postcoll=imagebulkpot_arr(qt+delq,mp['rp'],zp,mp)

#find number of collisions to get to tangent
#charge=0.0
#ncoll=0
#while(charge < qt):
#    dq=contact_charge(charge,vp,mp)
#    charge=charge+dq
#    ncoll+=1

#print("charge,ncoll",charge,ncoll)

np.savetxt("pascurve.dat",np.transpose(np.vstack((zp,Vbr))),delimiter="  ")
plt.figure()
plt.plot(zp,Vbr,color="black",linestyle="dashed",label="Paschen curve",linewidth=3)
#plt.plot(zp[int(0.625*len(zp)):],Vimg_tangent[int(0.625*len(zp)):],color="blue",linestyle="dotted",label="tangent img",linewidth=3)
plt.plot(zp,Vimg_tangent,color="blue",linestyle="dotted",label="tangent img",linewidth=3)
#plt.plot(zp[0:int(3.75*len(zp)/10)],Vimg_postcoll[0:int(3.75*len(zp)/10)],color="red",label="post-coll",linewidth=3)
plt.plot(zp,Vimg_postcoll,color="red",label="post-coll",linewidth=3)
#plt.plot(zp,Vtot_tangent,color="blue",label="tangent total")

fig,ax=plt.subplots(2,2,figsize=(8,8))
#plasma solve
#dq=contact_charge(charge,vp,mp)
print("solving intersect")
z1=fsolve(solve_intersect, [1.1*mp['dc_SI']], \
        args=(qt+delq,mp))[0]

#(intersectflag,zint)=intersect_solve_graphical(qt+dq,mp)
#if(intersectflag==False):
#    print("does not intersect")
#    sys.exit()

#print("z1,zint:",z1,zint)
#z1=min(zint)

z2=tloc
Npts_z=1000
z1z2=np.linspace(z1,z2,Npts_z)
qrelax=np.zeros(Npts_z)
Te=np.zeros(Npts_z)+evtemp
Te[0]=fsolve(solve_Te, [evtemp*10.0],args=(z1z2[0],mp))[0]
ne_init=1e12
nex_init=1e12
ndiss_init=1e12
ne=np.zeros(Npts_z-1)+ne_init
nex=np.zeros(Npts_z)+nex_init
ndiss=np.zeros(Npts_z)+ndiss_init
vf=vp*mp['e_rest']
dz=z1z2[1]-z1z2[0]
print("dz,qin,dq,z1:",dz,qt,delq,z1)
q1=qt+delq
q2=qt
qrelax[0]=q1

area_scaling=1.0
discharge_area=get_acoll(vp,mp)
max_area=np.pi*(mp['rp']**2)
max_area_scaling=max_area/discharge_area

print("max area scaling:",max_area_scaling)
if(area_scaling < max_area_scaling):
    discharge_area*=area_scaling
else:
    print("area scaling greater than max value of %f"%(max_area_scaling))
    sys.exit(0)

for i in range(Npts_z-1):
    #print(z1z2[i])
    #we are solving what happens between i and i+1
    Te[i+1]=fsolve(solve_Te, [evtemp*10.0],args=(z1z2[i+1],mp))[0]
    Te_avg=0.5*(Te[i]+Te[i+1])
    rate_ionize=arrh_rate(mp['Aiz'],mp['alphaiz'],mp['Ta_iz'],Te_avg)
    rate_diss=arrh_rate(mp['Ad'],mp['alphad'],mp['Ta_d'],Te_avg)
    rate_ex=arrh_rate(mp['Aex'],mp['alphaex'],mp['Ta_ex'],Te_avg)

    #print("Te,rate1,rate2:",Te[i],rate_ionize,rate_ex)
    cbar=np.sqrt(8.0*kboltz*Te_avg/np.pi/melec)
    
    #radfactor=1.0
    #discharge_vol=np.pi*(mp['rp']**2)*z1z2[i]/radfactor**2
    #discharge_area=np.pi*(mp['rp']**2)/radfactor**2
    
    discharge_vol=discharge_area*z1z2[i]
    
    q2=fsolve(solve_intersect_q, [qt], \
        args=(z1z2[i+1],mp))[0]
    delEdt=1.0/(8.0*np.pi*eps0*mp['rp'])*(q1**2-q2**2)/dz*vf
    
    #high pressure ambipolar solution
    Da=mp['D_i']*(1.0+Te_avg/mp['Temp'])
    Gama_by_n0=Da*np.pi/z1z2[i+1] #Da times beta (pi/l)
    Eloss_inel=rate_ionize*mp["NG"]*mp['Eiz']*echarge*discharge_vol\
            +rate_diss*mp["NG"]*mp['Ed']*echarge*discharge_vol\
            +rate_ex*mp["NG"]*mp['Eex']*echarge*discharge_vol
    Eloss_el=1.5*kboltz*(Te_avg-mp['Temp'])*(2.0*melec)/mp['m_gas']*collfreq(Te_avg,mp)*discharge_vol
    Eloss_w=(2*kboltz*Te_avg)*2.0*Gama_by_n0*discharge_area
    Eloss=Eloss_inel+Eloss_w+Eloss_el

    ne[i]=mp["EJfrac"]*delEdt/Eloss
    ndiss[i+1]=ndiss[i]+mp["dissmoles"]*rate_diss*mp['NG']*ne[i]/vf*dz
    nex[i+1]=nex[i]+rate_ex*mp['NG']*ne[i]/vf*dz
    qrelax[i+1]=q2
    q1=np.copy(q2)

ax[0][0].plot(z1z2[:],Te[:],linewidth=2)
print("Te max/min:",np.max(Te),np.min(Te))

#ax[0][1].set_xscale("log")
ax[0][1].set_yscale("log")
ax[0][1].plot(0.5*(z1z2[1:]+z1z2[0:-1]),ne,'r-',linewidth=3)
print("ne max/min:",np.max(ne),np.min(ne))

#ax[1][0].set_xscale("log")
ax[1][0].set_yscale("log")
#ax[1][0].set_ylim(1e12,1e22)
ax[1][0].plot(z1z2,ndiss,linewidth=3,label="dissociated")
ax[1][0].plot(z1z2,nex,linewidth=3,label="excited")
ax[1][0].legend()
print("ndiss max/min:",np.max(ndiss),np.min(ndiss))
print("nex max/min:",np.max(nex),np.min(nex))

#plt.xscale("log")
#plt.yscale("log")
ax[1][1].plot(z1z2,qrelax,linewidth=3)
print("qrelax max/min:",np.max(qrelax),np.min(qrelax))
    
np.savetxt("spec_single.dat",np.transpose(np.vstack((z1z2,z1z2/np.max(z1z2),nex,ndiss,qrelax))),delimiter="  ")

plt.figure()
plt.plot(0.5*(z1z2[1:]+z1z2[0:-1]),ne,'r*')
plt.show()
