import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from model_parameters import *
from electrostatics import *
from nonlinsolvers import *
from distfuncs import *
from matplotlib.collections import LineCollection

#main
font={'family':'Helvetica', 'size':'15'}
mpl.rc('font',**font)
mpl.rc('xtick',labelsize=15)
mpl.rc('ytick',labelsize=15)

grantemp=500.0 #m2/s2

mp=setmodelparams()
mp['bulkpot']=1
veldist_avg=np.sqrt(8.0*grantemp/np.pi)
Npartitions=10
velparts=np.linspace(0.0,\
        3.0*veldist_avg,Npartitions+1)
vpmidarr=0.5*(velparts[0:-1]+velparts[1:])

cdist=collidingdist(velparts,grantemp)
print("cdist:",cdist)
print("gaussdist integral:",np.trapz(cdist,velparts))

Nparts=np.zeros(Npartitions)
for pt in range(Npartitions):
    Nparts[pt]=mp['np']*collidingdist(0.5*(velparts[pt]+velparts[pt+1]),\
            grantemp)

qpart=np.zeros(Npartitions)

(qt,tloc)=tangent_solve_bisection(0.0,60e-9,mp,N=10000)
print("qt,tloc",qt,tloc)

N=1000000
max_z_log10=np.log10(5000.0*mp['dc_SI'])
min_z_log10=np.log10(0.37*mp['dc_SI'])
zp=np.logspace(min_z_log10,max_z_log10,N)
Vbr=paschen_arr(mp,zp)
Vimg_tangent,Vtot_tangent=imagebulkpot_arr(qt,mp['rp'],zp,mp)

delq=np.zeros(Npartitions)
for pt in range(Npartitions):
    vp=0.5*(velparts[pt]+velparts[pt+1])
    delq[pt]=contact_charge(qt,vp,mp)
    print("pt, delq_pt:",pt, delq[pt], Nparts[pt],Nparts[pt]*delq[pt])

    #find number of collisions to get to tangent
    #charge=0.0
    #ncoll=0
    #find number of collisions to get to tangent
    #while(charge < qt):
    #    dq=contact_charge(charge,vp,mp)
    #    charge=charge+dq
    #    ncoll+=1

    #print("pt, charge,ncoll",pt,charge,ncoll)


np.savetxt("nparts_delq.dat",np.transpose(np.vstack((0.5*(velparts[0:-1]+velparts[1:]),Nparts,delq))),delimiter="  ")
plt.figure()
plt.plot(zp,Vbr,color="black",linestyle="dashed",label="Paschen curve")
plt.plot(zp,Vimg_tangent,color="blue",linestyle="dotted",label="tangent img")
plt.plot(zp,Vtot_tangent,color="blue",label="tangent total")
plt.legend()

fig,ax=plt.subplots(2,2)
colormap = plt.cm.gist_ncar
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, Npartitions))))

Npts_z=1000
z1=np.zeros(Npartitions)
qrelax=np.zeros((Npartitions,Npts_z))
ne_init=1e12
nex_init=1e12
Te=np.zeros((Npartitions,Npts_z))+evtemp
ne=np.zeros((Npartitions,Npts_z-1))+ne_init
nex=np.zeros((Npartitions,Npts_z))+nex_init



for pt in range(Npartitions):
    lineset=[]
    vp=0.5*(velparts[pt]+velparts[pt+1])
    dq=delq[pt]
    #print("solving intersect,dq:",dq)
    #z1=fsolve(solve_intersect, [1.1*mp['dc_SI']], \
    #        args=(qt+dq,mp))[0]
    (intersectflag,zint)=intersect_solve_graphical(qt+dq,mp)
    if(intersectflag==False):
        print("does not intersect")
        sys.exit()

    z1[pt]=min(zint)
    z2=tloc
    #print("z1,z2:",z1,z2)
    z1z2=np.linspace(z1[pt],z2,Npts_z)
    Te[pt][0]=fsolve(solve_Te, [evtemp*10.0],args=(mp,z1z2[0]))[0]
    vf=vp*mp['e_rest']
    dz=z1z2[1]-z1z2[0]
    #print("dz,qin,dq,z1,z2:",dz,qt,dq,z1[pt],z2)
    q1=qt+dq
    q2=qt
    qrelax[pt][0]=q1
    for i in range(Npts_z-1):
        #print(z1z2[i])
        #we are solving what happens between i and i+1
        Te[pt][i+1]=fsolve(solve_Te, [evtemp*1.0],args=(mp,z1z2[i+1]))[0]
        Te_avg=0.5*(Te[pt][i]+Te[pt][i+1])
        rate_ionize=arrh_rate(mp['Aiz'],mp['alphaiz'],mp['Ta_iz'],Te_avg)
        rate_diss=arrh_rate(mp['Ad'],mp['alphad'],mp['Ta_d'],Te_avg)
        
        cbar=np.sqrt(8.0*kboltz*Te_avg/np.pi/melec)
        radfactor=1.0
        vol=np.pi*(mp['rp']**2)*z1z2[i]/radfactor**2
        area=np.pi*(mp['rp']**2)/radfactor**2
        q2=fsolve(solve_intersect_q, [qt], \
        args=(z1z2[i+1],mp))[0]
        delEdt=1.0/(8.0*np.pi*eps0*mp['rp'])*(q1**2-q2**2)/dz*vf
    
        #high pressure ambipolar solution
        Da=mp['D_i']*(1.0+Te_avg/mp['Temp'])
        Gama_by_n0=Da*np.pi/z1z2[i+1]
        Eloss_c=rate_ionize*mp["NG"]*mp['Eiz']*echarge*vol+rate_diss*mp["NG"]*mp['Ed']*echarge*vol
        Eloss_e=(2*kboltz*Te_avg)*2.0*Gama_by_n0*area
        Eloss=Eloss_c+Eloss_e
        
        #Eloss=rate_ionize*mp["NG"]*mp['Ei']*echarge*vol+rate_ex*mp["NG"]*mp['Eex']*echarge*vol+0.5*cbar*area*(2*kboltz*Te_avg)
        #print("Te,rate1,rate2,Eloss:",Te[pt][i],rate_ionize,rate_ex,Eloss)
        #uB=np.sqrt(kboltz*Te_avg/mp['m_ion'])
        #print("Eloss before:",Eloss)
        #Eloss += uB*area*(5.0*kboltz*Te_avg) 
        #print("Eloss after:",Eloss)
        #Eloss=rate_ionize*mp["NG"]*mp['Ei']*echarge
        #+rate_ex*mp["       NG"]*mp['Eex']*echarge+2.5*kboltz*Te[i]
        #Eloss=Eloss*uB*area
        
        ne[pt][i]=mp["EJfrac"]*delEdt/Eloss
        nex[pt][i+1]=nex[pt][i]+mp['dissmoles']*rate_diss*mp['NG']*ne[pt][i]/vf*dz
        qrelax[pt][i+1]=q2
        q1=np.copy(q2)
        
    print("vp,max/min ne,max/min nex,max/min Te:",\
            vp, np.max(ne[pt][:]),np.min(ne[pt][:]), \
            np.max(nex[pt][:]), np.min(nex[pt][:]), \
            np.max(Te[pt][:]),np.min(Te[pt][:]))

    ax[0][0].plot(z1z2,Te[pt][:],linewidth=2)

    ax[0][1].set_xscale("log")
    ax[0][1].set_yscale("log")
    ax[0][1].plot(0.5*(z1z2[1:]+z1z2[0:-1]),ne[pt][:],linewidth=2)
    tmparr=np.transpose(np.vstack((0.5*(z1z2[1:]+z1z2[0:-1]),ne[pt][:])))
 
    lineset.append(tmparr)

    ax[1][0].set_xscale("log")
    ax[1][0].set_yscale("log")
    #ax[1][0].set_ylim(1e12,1e22)
    ax[1][0].plot(z1z2,nex[pt][:],linewidth=2)

    #plt.xscale("log")
    #plt.yscale("log")
    ax[1][1].plot(z1z2,qrelax[pt][:],linewidth=2)

'''line_segments = LineCollection(lineset,linewidths=(0.5, 1, 1.5, 2),cmap='coolwarm',linestyles='solid')
line_segments.set_array(vpmidarr)
ax1.add_collection(line_segments)
axcb = fig.colorbar(line_segments)
axcb.set_label('velocity (m/s)')
plt.sci(line_segments)  # This allows interactive changing of the colormap.'''

plt.show()
