import scipy
import numpy as np
from scipy.optimize import fsolve
from constants import *
from scipy.interpolate import interp1d

def setmodelparams():
    modelParams = {}
    modelParams['pres'] =  101325.0 #Pa

    #separation distance
    #Ali, F. Sharmene, et al. "Charge exchange model of a disperse 
    #system of spherical powder particles." Conference Record of 1998 
    #IEEE Industry Applications Conference. 
    #Thirty-Third IAS Annual Meeting (Cat. No. 98CH36242). Vol. 3. IEEE, 1998.
    #modelParams['delc']  = 100e-9 # 100 nm

    #Tan, Zhen, Libin Zhang, and Zhonghua Yu. "Valuations of charging 
    #efficiency and critical gap for electron tunnelling 
    #under contact electrification model." Powder Technology 415 (2023): 118140.
    #modelParams['delc']  = 88.54e-9 # 100 nm
   
    #Ray, M., et al. "Eulerian modeling of charge transport in bi-disperse 
    #particulate flows due to triboelectrification." 
    #Physics of Fluids 32.2 (2020).
    #modelParams['delc']  = 1e-9 # 1 nm

    #Tan, Zhen, Libin Zhang, and Zhonghua Yu. "Valuations of charging 
    #efficiency and critical gap for electron tunnelling 
    #under contact electrification model." Powder Technology 415 (2023): 118140.
    #modelParams['delc']  = 10e-9 # 1 nm
    #modelParams['k_c']  = 0.1
   
    #using this for N2
    modelParams['delc']  = 88.54e-9 # 1 nm
    modelParams['k_c']  = 1.0
    
    #using this for CO2
    #modelParams['delc']  = 10e-9 # 1 nm
    #modelParams['k_c']  = 0.1
    
    #Theory of insulator charging
    #L.B. Schein, M. La~Ha and D. Novotny
    #PhysicsLettersA167(1992) 79-83
    #modelParams['delc']  = 10e-10 # 1 nm

    #PTFE particles
    #Investigating the Influence of Friction and Material Wear 
    #on Triboelectric Charge Transfer in Metal–Polymer Contacts
    #J. L. Armitage1 · A. Ghanbarzadeh1 · M. G. Bryant1 · A. Neville1
    #Tribology Letters (2022) 70:46
    #PTFE particle
    modelParams['E_part'] =  0.575e9 #Pa
    modelParams['G_part'] = 0.23e9 #Pa
    #E=2G(1+nu)
    modelParams['nu_part'] = 0.5*modelParams['E_part']/modelParams['G_part']-1.0
    modelParams['rho_part'] =2200.0 #kg/m3
    
    #Aluminium wall
    modelParams['E_wall'] =  68.9e9 #Pa
    modelParams['G_wall'] =  27.0e9 #Pa
    #E=2G(1+nu)
    modelParams['nu_wall'] = 0.5*modelParams['E_wall']/modelParams['G_wall']-1.0
    modelParams['rho_wall'] =2700.0 #kg/m3

    modelParams['rp']=0.0015 #particle radius (1 mm)
    modelParams['cht']=0.15 #channel size (25 cm)
    
    # bulk potential on?
    modelParams['bulkpot']=0
    
    # solids volume fraction
    modelParams['solidsvfrac']=0.01 #dilute granular flow

    #Paschen breakdown in N2 atmosphere, Raizer's texbook
    #fit of N2 curve
    modelParams['Vmin'] =  240.0 #V
    modelParams['pdc']  = 4.9 #mmTorr
    
    #Paschen breakdown in CO2 atmosphere, AIAA paper, 2013
    #modelParams['Vmin'] =  540.0 #V
    #modelParams['pdc']  = 5.0 #mmTorr
    
    #Investigating the Influence of Friction and Material 
    #Wear on Triboelectric Charge Transfer in Metal–Polymer Contacts
    #Tribology Letters (2022) 70:46
    #work function for Al (4.26 eV) vs PTFE (5.80 eV)
    modelParams['delphi']=1.54 #eV
    #work function for Cd (4.08 eV) vs PTFE (5.80 eV)
    #modelParams['delphi']=1.72 #eV

    #see Electrification of an elastic sphere by repeated impacts on a metal plate
    # Shuji Matsusaka, Mojtaba Ghadiri and Hiroaki Masuda
    #J. Phys. D: Appl. Phys. 33 (2000) 2311–2319
    #elastic parameter from Hertzian model
    modelParams['el_p']=(1.0-modelParams['nu_wall']**2)/modelParams['E_wall']+\
            (1.0-modelParams['nu_part']**2)/modelParams['E_part']
    modelParams['Temp']=300 #K
    modelParams['NG']=modelParams['pres']/kboltz/modelParams['Temp']
    modelParams['e_rest']=0.9

    print("poisson's ratio:",modelParams['nu_wall'],modelParams['nu_part'])

    #electron joule heating factor
    modelParams['EJfrac']=0.06 #from Thomas's MPT paper

    #an averaged value for N2chem from Yuan/Raja paper
    modelParams['Eiz']=15.6 #ionization energy (eV)
    modelParams['Ed']=9.757 #dissociation energy (eV)
    modelParams['alphaiz']=-0.3
    modelParams['alphad']=-0.7
    modelParams['Aiz']=4.483e-13
    modelParams['Ad']=1.959e-13
    modelParams['Ta_iz']=1.81e5 #K
    modelParams['Ta_d']=1.132e5 #K
    modelParams['m_ion']=28.0*mprot #N2+ ion
    modelParams['m_gas']=28.0*mprot #N2
    #All vibrational states excitation from
    #Zuowei et al 2014 Plasma Sci. Technol. 16 335
    #Simulation of Capacitively Coupled Dual- Frequency 
    #N2, O2, N2/O2 Discharges: Effects of External 
    #Parameters on Plasma Characteristics
    #this is for N2(A)
    modelParams['Eex']=6.17 #Vibrational excitation (eV)
    modelParams['alphaex']=0.0
    modelParams['Aex']=4.05e-15
    modelParams['Ta_ex']=5.36*evtemp
    modelParams['dissmoles']=2.0
    collfreq_data=np.loadtxt("meanenrg_vs_collfreqbyN_N2plasma")
    #this is for N2(a')
    #modelParams['Eex']=8.4 #Vibrational excitation (eV)
    #modelParams['alphaex']=0.0
    #modelParams['Aex']=1.47e-15
    #modelParams['Ta_ex']=8.79*evtemp
    
    #an averaged value from co2chem at std conditions
    #fit to BOLSIG run
    '''modelParams['Ei']=13.3 #ionization energy (eV)
    modelParams['Eiz']=13.3 #ionization energy (eV)
    modelParams['Ed']=4.85 #dissociation energy (eV)
    modelParams['alphaiz']=0.0
    modelParams['alphad']=0.0
    modelParams['Aiz']=1.6e-13
    modelParams['Ad']=1.0e-17
    modelParams['Ta_iz']=185672.0 #K
    modelParams['Ta_d']=34813.51 #K
    modelParams['m_ion']=44.0*mprot #CO2+ ion
    modelParams['m_gas']=44.0*mprot #CO2
    modelParams['Eex']=6.17 
    modelParams['alphaex']=0.0
    modelParams['Aex']=0.0
    modelParams['Ta_ex']=62200.15
    modelParams['dissmoles']=1.0
    collfreq_data=np.loadtxt("meanenrg_vs_collfreqbyN_CO2plasma")'''

    modelParams["collfreq_interp"]=interp1d(collfreq_data[:,0],collfreq_data[:,1],fill_value="extrapolate")

    modelParams['pdc_SI']=modelParams['pdc']*1e-3*101325.0/760.0 #Pa m
    modelParams['dc_SI']=modelParams['pdc_SI']/modelParams['pres']
    modelParams['B'] = modelParams['Vmin']/modelParams['pdc_SI']
    modelParams['C'] = 1.0-np.log(modelParams['pdc_SI'])
    
    #for CO2+ in CO2 from
    #Viehland and Mason, 
    #TRANSPORT PROPERTIES OF GASEOUS IONS OVER A WIDE ENERGY RANGE, IV
    #data given in cm2/V/s in this paper
    #K0=1.1627e-4 #m2/V/s #average of data from 50-300 Td
    
    #for N2+ in N2 at 310 K
    #Viehland and Mason, 
    #TRANSPORT PROPERTIES OF GASEOUS IONS OVER A WIDE ENERGY RANGE, IV
    #data given in cm2/V/s in this paper
    K0=2.25e-4 #m2/V/s #average of data from 50-300 Td
    
    N0=2.686763e25  #number density of ideal gas 0 C and 101.325 kPa
    N=modelParams['pres']/kboltz/modelParams['Temp']
    modelParams['mu_i']=K0*N0/N
    modelParams['D_i']=modelParams['mu_i']*modelParams["Temp"]/evtemp
    print("ion dcoeff:",modelParams['D_i'])
    
    return(modelParams)
