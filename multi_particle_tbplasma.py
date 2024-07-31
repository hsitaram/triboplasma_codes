import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from sys import argv
from model_parameters import *
from electrostatics import *
from nonlinsolvers import *
from distfuncs import *

#main
font={'family':'Helvetica', 'size':'15'}
mpl.rc('font',**font)
mpl.rc('xtick',labelsize=15)
mpl.rc('ytick',labelsize=15)

mp=setmodelparams()
mp['bulkpot']=1
vp=float(argv[1])
mp['solidsvfrac']=float(argv[2])
numberden=mp['solidsvfrac']/(4/3*np.pi*mp['rp']**3) 
print("particle number density:",numberden)
mp['rp']=float(argv[3])
mp['cht']=float(argv[4])


(qt,tloc)=tangent_solve_bisection(0.0,10e-9,mp,N=100000)
print("qt,tloc",qt,tloc)
qf2_z2=fsolve(solve_tangent, [qt,tloc], \
        args=(mp),xtol=1e-12)
print("qf2,z2",qf2_z2[0],qf2_z2[1])
print("error:",solve_tangent(qf2_z2,mp))

N=100000
max_z_log10=np.log10(5000.0*mp['dc_SI'])
min_z_log10=np.log10(0.37*mp['dc_SI'])
zp=np.logspace(min_z_log10,max_z_log10,N)
Vbr=paschen_arr(mp,zp)
Vimg_tangent,Vtot_tangent=imagebulkpot_arr(qt,mp['rp'],zp,mp)

delq=contact_charge(qt,vp,mp)
print("delq:",delq)
if(delq<0):
    print("contact charge less than 0")
    sys.exit()

Vimg_postcoll,Vtot_postcoll=imagebulkpot_arr(qt+delq,mp['rp'],zp,mp)

#find number of collisions to get to tangent
'''charge=0.0
ncoll=0
while(charge < qt):
    dq=contact_charge(charge,vp,mp)
    charge=charge+dq
    ncoll+=1

print("charge,ncoll",charge,ncoll)'''


plt.figure()
plt.xlabel("Distance (m)")
plt.ylabel("Potential (V)")
plt.plot(zp,Vbr,color="black",linestyle="dashed",label="Paschen curve",linewidth=3)
plt.plot(zp,Vimg_tangent,color="blue",linestyle="dashdot",label="$V_{img}(q_{f2},z)$",linewidth=3)
plt.plot(zp,Vtot_tangent,color="blue",label="$V_{img}(q_{f2},z)+V_b(q_{f2},z)$")
#plt.plot(zp,Vimg_postcoll,color="red",label="post-coll image")
plt.plot(zp,Vtot_postcoll,color="red",label="post-collision")
plt.legend()

fig,ax=plt.subplots(2,2)
#plasma solve
print("solving intersect")
#z1=fsolve(solve_intersect, [1.1*mp['dc_SI']], \
#        args=(qt+dq,mp))[0]

(intersectflag,zint)=intersect_solve_graphical(qt+delq,mp)
if(intersectflag==False):
    print("does not intersect")
    sys.exit()

#print("z1,zint:",z1,zint)
z1=min(zint)

z2=tloc
Npts_z=1000
z1z2=np.linspace(z1,z2,Npts_z)
qrelax=np.zeros(Npts_z)
Te=np.zeros(Npts_z)+evtemp
Te[0]=fsolve(solve_Te, [evtemp*10.0],args=(mp,z1z2[0]))[0]
ne_init=1e12
nex_init=1e12
ne=np.zeros(Npts_z-1)+ne_init
nex=np.zeros(Npts_z)+nex_init
vf=vp*mp['e_rest']
dz=z1z2[1]-z1z2[0]
print("dz,qin,delq,z1:",dz,qt,delq,z1)
q1=qt+delq
q2=qt
V1=paschen(mp['B'],mp['C'],mp['pres'],z1z2[0])
V2=V1
qrelax[0]=q1
for i in range(Npts_z-1):
    #print(z1z2[i])
    #we are solving what happens between i and i+1
    Te[i+1]=fsolve(solve_Te, [evtemp*10.0],args=(mp,z1z2[i+1]))[0]
    Te_avg=0.5*(Te[i]+Te[i+1])
    rate_ionize=arrh_rate(mp['Aiz'],mp['alphaiz'],mp['Ta_iz'],Te_avg)
    rate_diss=arrh_rate(mp['Ad'],mp['alphad'],mp['Ta_d'],Te_avg)
    
    #print("Te,rate1,rate2:",Te[i],rate_ionize,rate_ex)
    cbar=np.sqrt(8.0*kboltz*Te_avg/np.pi/melec)
    radfactor=1.0
    vol=np.pi*(mp['rp']**2)*z1z2[i]/radfactor**2
    area=np.pi*(mp['rp']**2)/radfactor**2
    q2=fsolve(solve_intersect_q, [qt], \
        args=(z1z2[i+1],mp))[0]
    V2=paschen(mp['B'],mp['C'],mp['pres'],z1z2[i+1])
    delEdt=1.0/(8.0*np.pi*eps0*mp['rp'])*(q1**2-q2**2)/dz*vf
    
    #high pressure ambipolar solution
    Da=mp['D_i']*(1.0+Te_avg/mp['Temp'])
    Gama_by_n0=Da*np.pi/z1z2[i+1]
    Eloss_c=rate_ionize*mp["NG"]*mp['Eiz']*echarge*vol+rate_diss*mp["NG"]*mp['Ed']*echarge*vol
    Eloss_e=(2*kboltz*Te_avg)*2.0*Gama_by_n0*area
    Eloss=Eloss_c+Eloss_e
    
    ne[i]=mp["EJfrac"]*delEdt/Eloss
    nex[i+1]=nex[i]+mp['dissmoles']*rate_diss*mp['NG']*ne[i]/vf*dz
    qrelax[i+1]=q2
    q1=np.copy(q2)

ax[0][0].set_title("Electron temperature (K)")
ax[0][0].plot(z1z2[:],Te[:],linewidth=3)
print("Te max/min:",np.max(Te),np.min(Te))

ax[0][1].set_title("Electron density (#/m3)")
#ax[0][1].set_xscale("log")
ax[0][1].set_yscale("log")
ax[0][1].plot(0.5*(z1z2[1:]+z1z2[0:-1]),ne,'r-',linewidth=3)
print("ne max/min:",np.max(ne),np.min(ne))

ax[1][0].set_title("N density (#/m3)")
#ax[1][0].set_xscale("log")
ax[1][0].set_yscale("log")
#ax[1][0].set_ylim(1e12,1e22)
ax[1][0].plot(z1z2,nex,linewidth=3)
ax[1][0].set_xlabel("Distance (m)")
print("nex max/min:",np.max(nex),np.min(nex))

#plt.xscale("log")
#plt.yscale("log")
ax[1][1].set_title("Particle charge (nC)")
ax[1][1].plot(z1z2,qrelax*1e9,linewidth=3)
ax[1][1].set_xlabel("Distance (m)")
print("qrelax max/min:",np.max(qrelax),np.min(qrelax))
plt.tight_layout()

np.savetxt("spec_mult.dat",np.transpose(np.vstack((z1z2,z1z2/np.max(z1z2),nex))),delimiter="  ")

outfile=open("rundata_multi","a")
outfile.write("%e\t%e\t%e\t%e\t"%(vp,mp['solidsvfrac'],mp['rp'],mp['cht']))
outfile.write("%e\t%e\t"%(delq,qt))
outfile.write("%e\t%e\t"%(z1z2[0],z1z2[-1]))
outfile.write("%e\t%e\t%e\t%e\t%e\t%e\t"%(np.max(Te),np.mean(Te),np.max(ne),np.mean(ne),np.max(nex),np.mean(nex)))
outfile.write("%e\t%e\n"%(np.max(qrelax),np.min(qrelax)))
outfile.close()

plt.figure()
plt.plot(0.5*(z1z2[1:]+z1z2[0:-1]),ne,'r*')
plt.show()
