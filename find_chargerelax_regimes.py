import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from sys import argv
from model_parameters import *
from electrostatics import *
from nonlinsolvers import *
from distfuncs import *

def write_paraview_file_cartmesh(fname,dx,prob_lo,N,ncdata,ccdata):

    one=1
    outfile=open(fname,'w')
    outfile.write("<?xml version=\"1.0\"?>\n")
    outfile.write("<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
    outfile.write("<RectilinearGrid WholeExtent=\"%d\t%d\t%d\t%d\t%d\t%d\">\n"%(one,N[0],one,N[1],one,one))
    outfile.write("<Piece Extent=\"%d\t%d\t%d\t%d\t%d\t%d\">\n"%(one,N[0],one,N[1],one,one))

    outfile.write("<PointData>\n")
    n_ncdata=ncdata.shape[0]
    if(n_ncdata > 0):
        for ndataset in range(n_ncdata):
            outfile.write("<DataArray type=\"Float32\" Name=\"Point_data%d\" format=\"ascii\">\n"%(ndataset))

            for i in range(ncdata.shape[1]):
                outfile.write("%e "%(ncdata[ndataset][i]))
            outfile.write("\n</DataArray>\n")
    outfile.write("</PointData>\n")

    outfile.write("<CellData>\n")
    n_ccdata=ccdata.shape[0]
    if(n_ccdata > 0):
        for ndataset in range(n_ccdata):
            outfile.write("<DataArray type=\"Float32\" Name=\"Cell_data%d\" format=\"ascii\">\n"%(ndataset))
            for i in range(ccdata.shape[1]):
                outfile.write("%e "%(ccdata[ndataset][i]))
            outfile.write("\n</DataArray>\n")
    outfile.write("</CellData>\n")

    outfile.write("<Coordinates>\n")

    outfile.write("<DataArray type=\"Float32\" Name=\"X\"  format=\"ascii\">\n")
    for i in range(N[0]):
        outfile.write("%e\t"%(prob_lo[0]+i*dx[0]))
    outfile.write("\n</DataArray>\n")

    outfile.write("<DataArray type=\"Float32\" Name=\"Y\"  format=\"ascii\">\n")
    for i in range(N[1]):
        outfile.write("%e\t"%(prob_lo[1]+i*dx[1]))
    outfile.write("\n</DataArray>\n")

    outfile.write("<DataArray type=\"Float32\" Name=\"Z\"  format=\"ascii\">\n")
    outfile.write("%e\t"%(0.0))
    outfile.write("\n</DataArray>\n")

    outfile.write("</Coordinates>\n")
    outfile.write("</Piece>\n")
    outfile.write("</RectilinearGrid>\n")
    outfile.write("</VTKFile>")

    outfile.close()

#main
font={'family':'Helvetica', 'size':'15'}
mpl.rc('font',**font)
mpl.rc('xtick',labelsize=15)
mpl.rc('ytick',labelsize=15)

mp=setmodelparams()
mp['bulkpot']=1
vp=20.0
mp['solidsvfrac']=0.01
numberden=mp['solidsvfrac']/(4/3*np.pi*mp['rp']**3) 
print("particle number density:",numberden)
mp['rp']=0.0015
mp['cht']=0.15
mp['delphi']=1.0

pdsweep_min=float(argv[1])
pdsweep_max=float(argv[2])
Vmsweep_min=float(argv[3])
Vmsweep_max=float(argv[4])
Nsw=int(argv[5])
pdmin_arr1=np.linspace(pdsweep_min,pdsweep_max,Nsw)
Vmin_arr1=np.linspace(Vmsweep_min,Vmsweep_max,Nsw)

#pdmin_arr,Vmin_arr=np.meshgrid(pdmin_arr1,Vmin_arr1)
delq_arr=np.zeros(Nsw*Nsw)
qeqbm_arr=np.zeros(Nsw*Nsw)
qt_arr=np.zeros(Nsw*Nsw)
dx=np.array([pdmin_arr1[1]-pdmin_arr1[0],Vmin_arr1[1]-Vmin_arr1[0]])
plo=np.array([pdmin_arr1[0],Vmin_arr1[0]])

outfile=open("regimedata","w")

#Vmin
for j in range(Nsw):
    #pdmin
    for i in range(Nsw):
        mp['pdc']  = pdmin_arr1[i]
        mp['Vmin'] =  Vmin_arr1[j]
        mp['pdc_SI']=mp['pdc']*1e-3*101325.0/760.0 #Pa m
        mp['dc_SI']=mp['pdc_SI']/mp['pres']
        mp['B'] = mp['Vmin']/mp['pdc_SI']
        mp['C'] = 1.0-np.log(mp['pdc_SI'])

        (qt,tloc)=tangent_solve_bisection(0.0,10e-9,mp,N=100000)
        print("qt,tloc",qt,tloc)
        qf2_z2=fsolve(solve_tangent, [qt,tloc], \
            args=(mp),xtol=1e-12)
        print("qf2,z2",qf2_z2[0],qf2_z2[1])
        print("error:",solve_tangent(qf2_z2,mp))
        qt_arr[j*Nsw+i]=qf2_z2[0]

        delq_arr[j*Nsw+i]=contact_charge(qt,vp,mp)
        qeqbm_arr[j*Nsw+i]=eqbm_charge(vp,mp)
        outfile.write("%e\t%e\t%e\t%e\t%e\n"%(pdmin_arr1[i],Vmin_arr1[j],\
                delq_arr[j*Nsw+i],qeqbm_arr[j*Nsw+i],qt_arr[j*Nsw+i]))

write_paraview_file_cartmesh("regimedata.vtr",dx,plo,np.array([Nsw,Nsw]),np.array([delq_arr,qeqbm_arr,qt_arr]),np.array([]))
