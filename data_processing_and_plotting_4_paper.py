import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.font_manager as font_manager


# load data for band structure to calculate densities 
dataFe = np.loadtxt("./bandstructure_unfolded-path-SC333-k444-Fe-ecut100.txt")
dataPt = np.loadtxt("./bandstructure_unfolded-path-SC333-k444-Pt-ecut100.txt")
dataAl = np.loadtxt("./bandstructure_unfolded-path-SC333-k444-Al-ecut100.txt")
dataAg = np.loadtxt("./bandstructure_unfolded-path-SC333-k444-Ag-ecut100.txt")

# load data of DOS, partial DOS and projected DOS on each atom
# For the Fe doping
en_w_Fe, dos_w_Fe, idos_w_Fe = np.loadtxt('./data_DOS_FeSC333_k444_e100/PdCoO2_Fe_doping_dos.dat', unpack=True)
energydos_w_Fe, edos_w_Fe, pdos_w_Fe =  np.loadtxt('./data_DOS_FeSC333_k444_e100/PdCoO2_Fe_doping_pdos.dat.pdos_tot',unpack=True)
ePd_w_Fe, Pdedos_w_Fe = np.loadtxt('./data_DOS_FeSC333_k444_e100/Pd_PDOS.dat', unpack=True)
eCo_w_Fe, Coedos_w_Fe = np.loadtxt('./data_DOS_FeSC333_k444_e100/Co_PDOS.dat', unpack=True)
eO_w_Fe, Oedos_w_Fe = np.loadtxt('./data_DOS_FeSC333_k444_e100/O_PDOS.dat', unpack=True)
edop_w_Fe, dopedos_w_Fe = np.loadtxt('./data_DOS_FeSC333_k444_e100/Fe_PDOS.dat', unpack=True)

# For Pt doping
en_w_Pt, dos_w_Pt, idos_w_Pt = np.loadtxt('./data_DOS_PtSC333_k444_e100/PdCoO2_Pt_doping_dos.dat', unpack=True)
energydos_w_Pt, edos_w_Pt, pdos_w_Pt =  np.loadtxt('./data_DOS_PtSC333_k444_e100/PdCoO2_Pt_doping_pdos.dat.pdos_tot',unpack=True)
ePd_w_Pt, Pdedos_w_Pt = np.loadtxt('./data_DOS_PtSC333_k444_e100/Pd_PDOS.dat', unpack=True)
eCo_w_Pt, Coedos_w_Pt = np.loadtxt('./data_DOS_PtSC333_k444_e100/Co_PDOS.dat', unpack=True)
eO_w_Pt, Oedos_w_Pt = np.loadtxt('./data_DOS_PtSC333_k444_e100/O_PDOS.dat', unpack=True)
edop_w_Pt, dopedos_w_Pt = np.loadtxt('./data_DOS_PtSC333_k444_e100/Pt_PDOS.dat', unpack=True)

# For Al doping
en_w_Al, dos_w_Al, idos_w_Al = np.loadtxt('./data_DOS_AlSC333_k444_e100/PdCoO2_Al_doping_dos.dat', unpack=True)
energydos_w_Al, edos_w_Al, pdos_w_Al =  np.loadtxt('./data_DOS_AlSC333_k444_e100/PdCoO2_Al_doping_pdos.dat.pdos_tot',unpack=True)
ePd_w_Al, Pdedos_w_Al = np.loadtxt('./data_DOS_AlSC333_k444_e100/Pd_PDOS.dat',unpack=True)
eCo_w_Al, Coedos_w_Al = np.loadtxt('./data_DOS_AlSC333_k444_e100/Co_PDOS.dat',unpack=True)
eO_w_Al, Oedos_w_Al = np.loadtxt('./data_DOS_AlSC333_k444_e100/O_PDOS.dat',unpack=True)
edop_w_Al, dopedos_w_Al = np.loadtxt('./data_DOS_AlSC333_k444_e100/Al_PDOS.dat',unpack=True)

# For Ag doping
en_w_Ag, dos_w_Ag, idos_w_Ag = np.loadtxt('./data_DOS_AgSC333_k444_e100/PdCoO2_Ag_doping_dos.dat', unpack=True)
energydos_w_Ag, edos_w_Ag, pdos_w_Ag =  np.loadtxt('./data_DOS_AgSC333_k444_e100/PdCoO2_Ag_doping_pdos.dat.pdos_tot',unpack=True)
ePd_w_Ag, Pdedos_w_Ag = np.loadtxt('./data_DOS_AgSC333_k444_e100/Pd_PDOS.dat', unpack=True)
eCo_w_Ag, Coedos_w_Ag = np.loadtxt('./data_DOS_AgSC333_k444_e100/Co_PDOS.dat', unpack=True)
eO_w_Ag, Oedos_w_Ag = np.loadtxt('./data_DOS_AgSC333_k444_e100/O_PDOS.dat', unpack=True)
edop_w_Ag, dopedos_w_Ag = np.loadtxt('./data_DOS_AgSC333_k444_e100/Ag_PDOS.dat', unpack=True)


# function defined to calculate the density from the band structure data 
def calc_density_1(data,Ebbd , dE, efermi, smearing):
    """ This function calculated the density from the bandstructure data
    Input:
    data = loaded data file with bandstructure data
    Ebbd = The energy window within which we want the data to be plot
    dE = energy step size
    efermi = Fermi energy 
    smearing = smearing window
    Output:
    Density matrix and energy 
    """
    # upper bound of the energy
    Emin = -Ebbd + efermi 
    # lower bound of the energy
    Emax =  Ebbd + efermi
    energy=np.linspace(Emin,Emax,dE) 
    # data selected within bounds +/- a small smearing window
    data = data[(data[:,1]>=Emin-max(smearing*10,0.1))*(data[:,1]<=Emax+max(smearing*10,0.1))]
    # unique k-points in the data
    nk = np.unique(data[:, 0])
    # to initialize the density matrix for different k and E points. 
    densitymat =np.zeros((len(nk),dE),dtype=float)
    for k,E,w in data[:,:3]:
        # index of the minimum in the list 
        ik=np.argmin(abs(k-nk))
        # Guassian function for density
        densitymat[ik,:] += w*np.exp( -(energy-E)**2/(2*smearing**2))
    return densitymat, energy

###############################################################################
###############################################################################
###############################################################################

# Fermi energies for different doped samples of PdCoO2
EfermiFe = 14.5565 
EfermiPt = 14.6303 
EfermiAl = 14.4954
EfermiAg = 14.4875

# relevant parameters taken for the plots 
smearing=0.02
Energyboundary=1.5
dE=5000

# calculating density for each specific dopant data file 

# For Fe doping
density_w_Fe, energy_w_Fe = calc_density_1(dataFe, Energyboundary, dE, EfermiFe, smearing)
k1_w_Fe = np.unique(dataFe[:, 0])/0.34140624870604636  # the number is rescaling factor from supercell to primitive cell 
E1_w_Fe = energy_w_Fe-EfermiFe
k1_w_Fe,E1_w_Fe = np.meshgrid(k1_w_Fe,E1_w_Fe)

# For Pt doping
density_w_Pt, energy_w_Pt = calc_density_1(dataPt, Energyboundary, dE, EfermiPt, smearing) 
k1_w_Pt=np.unique(dataPt[:, 0])/0.34140624870604636 
E1_w_Pt=energy_w_Pt-EfermiPt
k1_w_Pt,E1_w_Pt=np.meshgrid(k1_w_Pt,E1_w_Pt)

# For Al doping
density_w_Al, energy_w_Al =  calc_density_1(dataAl, Energyboundary, dE, EfermiAl, smearing)
k1_w_Al=np.unique(dataAl[:, 0])/0.34140624870604636 
E1_w_Al=energy_w_Al-EfermiAl
k1_w_Al,E1_w_Al=np.meshgrid(k1_w_Al,E1_w_Al)

# For Ag doping
density_w_Ag, energy_w_Ag =  calc_density_1(dataAg, Energyboundary, dE, EfermiAg, smearing)
k1_w_Ag=np.unique(dataAg[:, 0])/0.34140624870604636 
E1_w_Ag=energy_w_Ag-EfermiAg
k1_w_Ag,E1_w_Ag=np.meshgrid(k1_w_Ag,E1_w_Ag)


# The following is to set the size of the figure and gridspec_kw is to adjust the size of the plots in the figure.
# for e.g. bandstructure is bigger and DOS is smaller 
font = font_manager.FontProperties(family='Arial',
                                   weight='normal',
                                   style='normal', size=8) #font property for legends

fig, axs = plt.subplots(4, 2,figsize=(10,8),gridspec_kw={'width_ratios':[1,0.3]} )

###############################################################################################################

# density plot for Pt doping
pcm00 = axs[0, 0].pcolormesh(k1_w_Pt,E1_w_Pt,density_w_Pt.T, cmap='magma', shading='auto', 
                             norm=colors.LogNorm(vmin= np.min(density_w_Pt.T)+0.01, vmax= np.max(density_w_Pt.T)))
axs[0,0].spines['top'].set_linewidth(1.5)
axs[0,0].spines['right'].set_linewidth(1.5)
axs[0,0].spines['bottom'].set_linewidth(1.5)
axs[0,0].spines['left'].set_linewidth(1.5)
fig.colorbar(pcm00,ax = axs[0,0],extend="neither",location='left')#,shrink=0.80)
axs[0,0].tick_params(axis='both',direction='in',colors='w',length=4,width=1.5)
axs[0,0].set_xticks(ticks= [0, 1.2581, 2.5162, 3.8095, 4.3284], labels=['$\Gamma$','L', 'F','$\Gamma$','T'])
axs[0,0].set_yticks(ticks= [-1,0,1],
                    labels=['-1','0', '1'],color='k',fontsize=14,fontname='Arial')
axs[0,0].set_ylabel('$\epsilon - \epsilon_F$ (eV)',fontsize=14,fontname='Arial')
axs[0,0].yaxis.set_label_coords(-.05, .5)
axs[0,0].axhline(0, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[0,0].axvline(1.2581, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[0,0].axvline(2.5162, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[0,0].axvline(3.8095, linewidth=1, color='w', alpha=0.8, linestyle = '-')

# DOS for Pt doping
axs[0,1].plot(dos_w_Pt/27,en_w_Pt-EfermiPt,linewidth = 1, color="k", label="Total")
axs[0,1].plot(Pdedos_w_Pt/26,ePd_w_Pt-EfermiPt,linewidth = 1, color="darkgoldenrod", label="Pd")
axs[0,1].plot(Coedos_w_Pt/27,eCo_w_Pt-EfermiPt,linewidth = 1, color="royalblue",label= "Co")
axs[0,1].plot(Oedos_w_Pt/54,eO_w_Pt-EfermiPt,linewidth = 1, color="r", label = "O")
axs[0,1].plot(dopedos_w_Pt,edop_w_Pt-EfermiPt,linewidth = 1, color="aqua", label = "Pt")
axs[0,1].spines['top'].set_linewidth(1.5)
axs[0,1].spines['right'].set_linewidth(1.5)
axs[0,1].spines['bottom'].set_linewidth(1.5)
axs[0,1].spines['left'].set_linewidth(1.5)
axs[0,1].set_xlim([0,14])  
axs[0,1].set_ylim([-1.5,1.5])
axs[0,1].set_xticks(ticks= [0,5,10],
                    labels=['0','5','10'],fontsize=14,fontname='Arial')
axs[0,1].tick_params(axis='x',direction='in',length=4,width=1.5,labelsize=14)
axs[0,1].yaxis.set_visible(False)
axs[0,1].axhline(0, linewidth=1, color='k', alpha=0.8, linestyle = '-')
axs[0,1].legend(loc='upper right',fontsize="8",prop=font,frameon=False,facecolor='none',borderpad=0.2,labelspacing=0.3,handlelength=1,handletextpad=0.4,borderaxespad=0.2,bbox_to_anchor=(1,0.6))

#############################################################################################################

# Density plot for Ag doping
pcm10 = axs[1, 0].pcolormesh(k1_w_Ag,E1_w_Ag,density_w_Ag.T, cmap='magma', shading='auto', 
                             norm=colors.LogNorm(vmin= np.min(density_w_Ag.T)+0.01, vmax= np.max(density_w_Ag.T)))
axs[1,0].spines['top'].set_linewidth(1.5)
axs[1,0].spines['right'].set_linewidth(1.5)
axs[1,0].spines['bottom'].set_linewidth(1.5)
axs[1,0].spines['left'].set_linewidth(1.5)
fig.colorbar(pcm10,ax = axs[1,0],extend="neither",location='left')#,shrink=0.80)
axs[1,0].tick_params(axis='both',direction='in',colors='w',length=4,width=1.5,labelsize=14)
axs[1,0].set_xticks(ticks= [0, 1.2581, 2.5162, 3.8095, 4.3284],
                    labels=['$\Gamma$','L', 'F','$\Gamma$','T'],color='k',fontsize=14,fontname='Arial')
axs[1,0].set_yticks(ticks= [-1,0,1],
                    labels=['-1','0', '1'],color='k',fontsize=14,fontname='Arial')
axs[1,0].set_ylabel('$\epsilon - \epsilon_F$ (eV)',fontsize=14,fontname='Arial')
axs[1,0].yaxis.set_label_coords(-.05, .5)
axs[1,0].axhline(0, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[1,0].axvline(1.2581, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[1,0].axvline(2.5162, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[1,0].axvline(3.8095, linewidth=1, color='w', alpha=0.8, linestyle = '-')

# DOS for Ag doping
# here we've divided the DOS for the number of the respective atoms in the Supercell. 
axs[1,1].plot(dos_w_Ag/27,en_w_Ag-EfermiAg,linewidth = 1, color="k", label="Total") #marker=".", markersize=4, 
axs[1,1].plot(Pdedos_w_Ag/26,ePd_w_Ag-EfermiAg,linewidth = 1, color="darkgoldenrod", label="Pd")
axs[1,1].plot(Coedos_w_Ag/27,eCo_w_Ag-EfermiAg,linewidth = 1, color="royalblue",label= "Co")
axs[1,1].plot(Oedos_w_Ag/54,eO_w_Ag-EfermiAg,linewidth = 1, color="r", label = "O")
axs[1,1].plot(dopedos_w_Ag,edop_w_Ag-EfermiAg,linewidth = 1, color="aqua", label = "Ag")
axs[1,1].spines['top'].set_linewidth(1.5)
axs[1,1].spines['right'].set_linewidth(1.5)
axs[1,1].spines['bottom'].set_linewidth(1.5)
axs[1,1].spines['left'].set_linewidth(1.5)
axs[1,1].set_xlim([0,14])  
axs[1,1].set_ylim([-1.5,1.5])
axs[1,1].set_xticks(ticks= [0,5,10],
                    labels=['0','5', '10'],fontsize=14,fontname='Arial')
axs[1,1].tick_params(axis='x',direction='in',length=4,width=1.5,labelsize=14)
axs[1,1].yaxis.set_visible(False)
axs[1,1].set_xlabel('DOS (states/eV atom)',fontsize=14,fontname='Arial')
axs[1,1].axhline(0, linewidth=1, color='k', alpha=0.8, linestyle = '-')
axs[1,1].legend(loc='upper right',fontsize="8",prop=font,frameon=False,facecolor='none',borderpad=0.2,labelspacing=0.3,handlelength=1,handletextpad=0.4,borderaxespad=0.2,bbox_to_anchor=(1,0.6))

##########################################################################################################

# density plot for Fe doping
pcm20 =axs[2, 0].pcolormesh(k1_w_Fe,E1_w_Fe,density_w_Fe.T, cmap='magma', shading='auto', 
                            norm=colors.LogNorm(vmin= np.min(density_w_Fe.T)+0.01, vmax= np.max(density_w_Fe.T)))
axs[2,0].spines['top'].set_linewidth(1.5)
axs[2,0].spines['right'].set_linewidth(1.5)
axs[2,0].spines['bottom'].set_linewidth(1.5)
axs[2,0].spines['left'].set_linewidth(1.5)
fig.colorbar(pcm20,ax = axs[2,0],extend="neither",location='left')#,shrink=0.80)
axs[2,0].tick_params(axis='both',direction='in',colors='w',length=4,width=1.5)
axs[2,0].set_xticks(ticks= [0, 1.2581, 2.5162, 3.8095, 4.3284], labels=['$\Gamma$','L', 'F','$\Gamma$','T'])
axs[2,0].set_yticks(ticks= [-1,0,1],
                    labels=['-1','0', '1'],color='k',fontsize=14,fontname='Arial')
axs[2,0].set_ylabel('$\epsilon - \epsilon_F$ (eV)',fontsize=14,fontname='Arial')
axs[2,0].yaxis.set_label_coords(-.05, .5)
axs[2,0].axhline(0, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[2,0].axvline(1.2581, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[2,0].axvline(2.5162, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[2,0].axvline(3.8095, linewidth=1, color='w', alpha=0.8, linestyle = '-')

# DOS plots for Fe doping
axs[2,1].plot(dos_w_Fe/27,en_w_Fe-EfermiFe,linewidth = 1, color="k", label="Total")
axs[2,1].plot(Pdedos_w_Fe/27,ePd_w_Fe-EfermiFe,linewidth = 1, color="darkgoldenrod", label="Pd")
axs[2,1].plot(Coedos_w_Fe/26,eCo_w_Fe-EfermiFe,linewidth = 1, color="royalblue",label= "Co")
axs[2,1].plot(Oedos_w_Fe/54,eO_w_Fe-EfermiFe,linewidth = 1, color="r", label = "O")
axs[2,1].plot(dopedos_w_Fe,edop_w_Fe-EfermiFe,linewidth = 1, color="aqua", label = "Fe")
axs[2,1].spines['top'].set_linewidth(1.5)
axs[2,1].spines['right'].set_linewidth(1.5)
axs[2,1].spines['bottom'].set_linewidth(1.5)
axs[2,1].spines['left'].set_linewidth(1.5)
axs[2,1].set_xlim([0,14])  
axs[2,1].set_ylim([-1.5,1.5])
axs[2,1].set_xticks(ticks= [0,5,10],
                    labels=['0','5','10'],fontsize=14,fontname='Arial')
axs[2,1].tick_params(axis='x',direction='in',length=4,width=1.5,labelsize=14)
axs[2,1].yaxis.set_visible(False)
axs[2,1].axhline(0, linewidth=1, color='k', alpha=0.8, linestyle = '-')
# bbox_to_anchor is basically to move the legends around.
axs[2,1].legend(loc='upper right',fontsize="8",prop=font,frameon=False,facecolor='none',borderpad=0.2,labelspacing=0.3,handlelength=1,handletextpad=0.4,borderaxespad=0.2,bbox_to_anchor=(1,0.6))

##############################################################################################################

# Density plot for Al doping
pcm30 = axs[3, 0].pcolormesh(k1_w_Al,E1_w_Al,density_w_Al.T, cmap='magma', shading='auto', 
                             norm=colors.LogNorm(vmin= np.min(density_w_Al.T)+0.01, vmax= np.max(density_w_Al.T)))
axs[3,0].spines['top'].set_linewidth(1.5)
axs[3,0].spines['right'].set_linewidth(1.5)
axs[3,0].spines['bottom'].set_linewidth(1.5)
axs[3,0].spines['left'].set_linewidth(1.5)
fig.colorbar(pcm30,ax = axs[3,0],extend="neither",location='left')#,shrink=0.80)
axs[3,0].tick_params(axis='both',direction='in',colors='w',length=4,width=1.5,labelsize=14)
axs[3,0].set_xticks(ticks= [0, 1.2581, 2.5162, 3.8095, 4.3284],
                    labels=['$\Gamma$','L', 'F','$\Gamma$','T'],color='k',fontsize=14,fontname='Arial')
axs[3,0].set_yticks(ticks= [-1,0,1],
                    labels=['-1','0', '1'],color='k',fontsize=14,fontname='Arial')
axs[3,0].set_ylabel('$\epsilon - \epsilon_F$ (eV)',fontsize=14,fontname='Arial')
axs[3,0].yaxis.set_label_coords(-.05, .5)
axs[3,0].axhline(0, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[3,0].axvline(1.2581, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[3,0].axvline(2.5162, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[3,0].axvline(3.8095, linewidth=1, color='w', alpha=0.8, linestyle = '-')

# DOS for Al doping
# here I've divided the DOS for the number of the respective atoms in the Supercell. 
axs[3,1].plot(dos_w_Al/27,en_w_Al-EfermiAl,linewidth = 1, color="k", label="Total") #marker=".", markersize=4, 
axs[3,1].plot(Pdedos_w_Al/27,ePd_w_Al-EfermiAl,linewidth = 1, color="darkgoldenrod", label="Pd")
axs[3,1].plot(Coedos_w_Al/26,eCo_w_Al-EfermiAl,linewidth = 1, color="royalblue",label= "Co")
axs[3,1].plot(Oedos_w_Al/54,eO_w_Al-EfermiAl,linewidth = 1, color="r", label = "O")
axs[3,1].plot(dopedos_w_Al,edop_w_Al-EfermiAl,linewidth = 1, color="aqua", label = "Al")
axs[3,1].spines['top'].set_linewidth(1.5)
axs[3,1].spines['right'].set_linewidth(1.5)
axs[3,1].spines['bottom'].set_linewidth(1.5)
axs[3,1].spines['left'].set_linewidth(1.5)
axs[3,1].set_xlim([0,14])  
axs[3,1].set_ylim([-1.5,1.5])
axs[3,1].set_xticks(ticks= [0,5,10],
                    labels=['0','5', '10'],fontsize=14,fontname='Arial')
axs[3,1].tick_params(axis='x',direction='in',length=4,width=1.5,labelsize=14)
axs[3,1].yaxis.set_visible(False)
axs[3,1].set_xlabel('DOS (states/eV atom)',fontsize=14,fontname='Arial')
axs[3,1].axhline(0, linewidth=1, color='k', alpha=0.8, linestyle = '-')
# bbox_to_anchor is basically to move the legends around--adjust it accordingly. 
axs[3,1].legend(loc='upper right',fontsize="8",prop=font,frameon=False,facecolor='none',borderpad=0.2,labelspacing=0.3,handlelength=1,handletextpad=0.4,borderaxespad=0.2,bbox_to_anchor=(1,0.6))

#######################################################################################################

# to adjust the spacing between the three plots both as width and height. 
plt.subplots_adjust(wspace=0.04, hspace=0.15)

plt.savefig("Figure_DOS_w_Ag.png", dpi = 1200) #change to dpi 1200 for publishing.
