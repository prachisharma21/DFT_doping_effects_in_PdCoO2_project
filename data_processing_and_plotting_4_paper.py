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
enFe, dosFe, idosFe = np.loadtxt('./data_DOS_FeSC333_k444_e100/PdCoO2_Fe_doping_dos.dat', unpack=True)
energydosFe, edosFe, pdosFe =  np.loadtxt('./data_DOS_FeSC333_k444_e100/PdCoO2_Fe_doping_pdos.dat.pdos_tot',unpack=True)
ePdFe, PdedosFe = np.loadtxt('./data_DOS_FeSC333_k444_e100/Pd_PDOS.dat', unpack=True)
eCoFe, CoedosFe = np.loadtxt('./data_DOS_FeSC333_k444_e100/Co_PDOS.dat', unpack=True)
eOFe, OedosFe = np.loadtxt('./data_DOS_FeSC333_k444_e100/O_PDOS.dat', unpack=True)
eFeFe, FeedosFe = np.loadtxt('./data_DOS_FeSC333_k444_e100/Fe_PDOS.dat', unpack=True)

# For Pt doping
enPt, dosPt, idosPt = np.loadtxt('./data_DOS_PtSC333_k444_e100/PdCoO2_Pt_doping_dos.dat', unpack=True)
energydosPt, edosPt, pdosPt =  np.loadtxt('./data_DOS_PtSC333_k444_e100/PdCoO2_Pt_doping_pdos.dat.pdos_tot',unpack=True)
ePdPt, PdedosPt = np.loadtxt('./data_DOS_PtSC333_k444_e100/Pd_PDOS.dat', unpack=True)
eCoPt, CoedosPt = np.loadtxt('./data_DOS_PtSC333_k444_e100/Co_PDOS.dat', unpack=True)
eOPt, OedosPt = np.loadtxt('./data_DOS_PtSC333_k444_e100/O_PDOS.dat', unpack=True)
eFePt, FeedosPt = np.loadtxt('./data_DOS_PtSC333_k444_e100/Pt_PDOS.dat', unpack=True)

# For Al doping
enAl, dosAl, idosAl = np.loadtxt('./data_DOS_AlSC333_k444_e100/PdCoO2_Al_doping_dos.dat', unpack=True)
energydosAl, edosAl, pdosAl =  np.loadtxt('./data_DOS_AlSC333_k444_e100/PdCoO2_Al_doping_pdos.dat.pdos_tot',unpack=True)
ePdAl, PdedosAl = np.loadtxt('./data_DOS_AlSC333_k444_e100/Pd_PDOS.dat',unpack=True)
eCoAl, CoedosAl = np.loadtxt('./data_DOS_AlSC333_k444_e100/Co_PDOS.dat',unpack=True)
eOAl, OedosAl = np.loadtxt('./data_DOS_AlSC333_k444_e100/O_PDOS.dat',unpack=True)
eFeAl, FeedosAl = np.loadtxt('./data_DOS_AlSC333_k444_e100/Al_PDOS.dat',unpack=True)

# For Ag doping
enAg, dosAg, idosAg = np.loadtxt('./data_DOS_AgSC333_k444_e100/PdCoO2_Ag_doping_dos.dat', unpack=True)
energydosAg, edosAg, pdosAg =  np.loadtxt('./data_DOS_AgSC333_k444_e100/PdCoO2_Ag_doping_pdos.dat.pdos_tot',unpack=True)
ePdAg, PdedosAg = np.loadtxt('./data_DOS_AgSC333_k444_e100/Pd_PDOS.dat', unpack=True)
eCoAg, CoedosAg = np.loadtxt('./data_DOS_AgSC333_k444_e100/Co_PDOS.dat', unpack=True)
eOAg, OedosAg = np.loadtxt('./data_DOS_AgSC333_k444_e100/O_PDOS.dat', unpack=True)
eFeAg, FeedosAg = np.loadtxt('./data_DOS_AgSC333_k444_e100/Ag_PDOS.dat', unpack=True)


# function defined to calculate the density from the band structure data 

def calc_density_1(data,Ebbd , dE, efermi, smearing):
    """ This function calculated the density from the bandstructure data
    input paramaters: 
    data = loaded data file with bandstructure data
    Ebbd = The energy window within which we want the data to be plot
    dE = energy step size
    efermi = Fermi energy 
    smearing = smearing window
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
densityFe, energyFe = calc_density_1(dataFe, Energyboundary, dE, EfermiFe, smearing)
k1Fe = np.unique(dataFe[:, 0])/0.34140624870604636  # the number is rescaling factor from supercell to primitive cell 
E1Fe = energyFe-EfermiFe
k1Fe,E1Fe = np.meshgrid(k1Fe,E1Fe)

# For Pt doping
densityPt, energyPt = calc_density_1(dataPt, Energyboundary, dE, EfermiPt, smearing) 
k1Pt=np.unique(dataPt[:, 0])/0.34140624870604636 
E1Pt=energyPt-EfermiPt
k1Pt,E1Pt=np.meshgrid(k1Pt,E1Pt)

# For Al doping
densityAl, energyAl =  calc_density_1(dataAl, Energyboundary, dE, EfermiAl, smearing)
k1Al=np.unique(dataAl[:, 0])/0.34140624870604636 
E1Al=energyAl-EfermiAl
k1Al,E1Al=np.meshgrid(k1Al,E1Al)

# For Ag doping
densityAg, energyAg =  calc_density_1(dataAg, Energyboundary, dE, EfermiAg, smearing)
k1Ag=np.unique(dataAg[:, 0])/0.34140624870604636 
E1Ag=energyAg-EfermiAg
k1Ag,E1Ag=np.meshgrid(k1Ag,E1Ag)


# The following is to set the size of the figure and gridspec_kw is to adjust the size of the plots in the figure.
# for e.g. bandstructure is bigger and DOS is smaller 
font = font_manager.FontProperties(family='Arial',
                                   weight='normal',
                                   style='normal', size=8) #font property for legends

fig, axs = plt.subplots(4, 2,figsize=(10,8),gridspec_kw={'width_ratios':[1,0.3]} )

# density pot for Fe doping
pcm00 =axs[2, 0].pcolormesh(k1Fe,E1Fe,densityFe.T, cmap='magma', shading='auto', 
                            norm=colors.LogNorm(vmin= np.min(densityFe.T)+0.01, vmax= np.max(densityFe.T)))
axs[2,0].spines['top'].set_linewidth(1.5)
axs[2,0].spines['right'].set_linewidth(1.5)
axs[2,0].spines['bottom'].set_linewidth(1.5)
axs[2,0].spines['left'].set_linewidth(1.5)
fig.colorbar(pcm00,ax = axs[2,0],extend="neither",location='left')#,shrink=0.80)
axs[2,0].tick_params(axis='both',direction='in',colors='w',length=4,width=1.5)
axs[2,0].set_xticks(ticks= [0, 1.2581, 2.5162, 3.8095, 4.3284], labels=[])
axs[2,0].set_yticks(ticks= [-1,0,1],
                    labels=['-1','0', '1'],color='k',fontsize=14,fontname='Arial')
axs[2,0].set_ylabel('$\epsilon - \epsilon_F$ (eV)',fontsize=14,fontname='Arial')
axs[2,0].yaxis.set_label_coords(-.05, .5)
axs[2,0].axhline(0, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[2,0].axvline(1.2581, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[2,0].axvline(2.5162, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[2,0].axvline(3.8095, linewidth=1, color='w', alpha=0.8, linestyle = '-')

# DOS plots for Fe doping
axs[2,1].plot(dosFe/27,enFe-EfermiFe,linewidth = 1, color="k", label="Total")
axs[2,1].plot(PdedosFe/27,ePdFe-EfermiFe,linewidth = 1, color="darkgoldenrod", label="Pd")
axs[2,1].plot(CoedosFe/26,eCoFe-EfermiFe,linewidth = 1, color="royalblue",label= "Co")
axs[2,1].plot(OedosFe/54,eOFe-EfermiFe,linewidth = 1, color="r", label = "O")
axs[2,1].plot(FeedosFe,eFeFe-EfermiFe,linewidth = 1, color="aqua", label = "Fe")
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

# axs[0,1].set_xlim([0,22])  
# axs[0,1].set_ylim([-1.5,1.5]) 
# # to hide the y-axis 
# axs[0,1].yaxis.set_visible(False)
# # the DOS are not divided for each atoms--in case you want to keep it like.Third plot is with divided to give you an idea
# # Need to check with Turan and Chris--dependeing upon that we can only put the label in the last plot and remove these 
# #axs[0,1].set_xlabel('DOS(states/(eV)',fontsize=8,fontname='Ubuntu')
# axs[0,1].xaxis.set_label_coords(1.3, -0.)
# axs[0,1].axhline(0, linewidth=0.8, color='lime', alpha=0.8, linestyle = '--')
# # bbox_to_anchor is basically to move the legends around--adjust it accordingly. 
# axs[0,1].legend(loc='upper right',fontsize="6",bbox_to_anchor=(1.55,1))


pcm10 = axs[0, 0].pcolormesh(k1Pt,E1Pt,densityPt.T, cmap='magma', shading='auto', 
                             norm=colors.LogNorm(vmin= np.min(densityPt.T)+0.01, vmax= np.max(densityPt.T)))
axs[0,0].spines['top'].set_linewidth(1.5)
axs[0,0].spines['right'].set_linewidth(1.5)
axs[0,0].spines['bottom'].set_linewidth(1.5)
axs[0,0].spines['left'].set_linewidth(1.5)
fig.colorbar(pcm10,ax = axs[0,0],extend="neither",location='left')#,shrink=0.80)
axs[0,0].tick_params(axis='both',direction='in',colors='w',length=4,width=1.5)
axs[0,0].set_xticks(ticks= [0, 1.2581, 2.5162, 3.8095, 4.3284], labels=[])
axs[0,0].set_yticks(ticks= [-1,0,1],
                    labels=['-1','0', '1'],color='k',fontsize=14,fontname='Arial')
axs[0,0].set_ylabel('$\epsilon - \epsilon_F$ (eV)',fontsize=14,fontname='Arial')
axs[0,0].yaxis.set_label_coords(-.05, .5)
axs[0,0].axhline(0, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[0,0].axvline(1.2581, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[0,0].axvline(2.5162, linewidth=1, color='w', alpha=0.8, linestyle = '-')
axs[0,0].axvline(3.8095, linewidth=1, color='w', alpha=0.8, linestyle = '-')

# # in case you need to shrink the colorbar--remove the comment from the next line and adjust the parameter
# fig.colorbar(pcm10,ax = axs[1,0],extend="both",location='left')#,shrink=0.80)
# axs[1,0].xaxis.set_visible(False)
# axs[1,0].set_xticks(ticks= [0, 1.2581, 2.5162, 3.8095, 4.3284])
# axs[1,0].set_ylabel('$\epsilon - \epsilon_F$ (eV)',fontsize=10,fontname='Arial')
# axs[1,0].yaxis.set_label_coords(-.05, .5)
# axs[1,0].axhline(0, linewidth=0.8, color='lime', alpha=0.8, linestyle = '--')

#dosFe/27,enFe
axs[0,1].plot(dosPt/27,enPt-EfermiPt,linewidth = 1, color="k", label="Total")
axs[0,1].plot(PdedosPt/26,ePdPt-EfermiPt,linewidth = 1, color="darkgoldenrod", label="Pd")
axs[0,1].plot(CoedosPt/27,eCoPt-EfermiPt,linewidth = 1, color="royalblue",label= "Co")
axs[0,1].plot(OedosPt/54,eOPt-EfermiPt,linewidth = 1, color="r", label = "O")
axs[0,1].plot(FeedosPt,eFePt-EfermiPt,linewidth = 1, color="aqua", label = "Pt")
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
#axs[1,1].set_xlabel('DOS (states/eV atom)',fontsize=14,fontname='Arial')
#axs[2,1].xaxis.set_label_coords(1.35, -0.)
axs[0,1].axhline(0, linewidth=1, color='k', alpha=0.8, linestyle = '-')
# bbox_to_anchor is basically to move the legends around--adjust it accordingly. 
axs[0,1].legend(loc='upper right',fontsize="8",prop=font,frameon=False,facecolor='none',borderpad=0.2,labelspacing=0.3,handlelength=1,handletextpad=0.4,borderaxespad=0.2,bbox_to_anchor=(1,0.6))

# axs[1,1].set_xlim([0,15])  
# axs[1,1].set_ylim([-1.5,1.5]) 
# axs[1,1].yaxis.set_visible(False)
# #axs[1,1].set_xlabel('DOS(states/(eV)',fontsize=8,fontname='Ubuntu')
# axs[1,1].xaxis.set_label_coords(1.3, -0.)
# axs[1,1].axhline(0, linewidth=0.8, color='lime', alpha=0.8, linestyle = '--')
# # bbox_to_anchor is basically to move the legends around--adjust it accordingly. 
# axs[1,1].legend(loc='upper right',fontsize="6",bbox_to_anchor=(1.55,1))


pcm20 = axs[3, 0].pcolormesh(k1Al,E1Al,densityAl.T, cmap='magma', shading='auto', 
                             norm=colors.LogNorm(vmin= np.min(densityAl.T)+0.01, vmax= np.max(densityAl.T)))
axs[3,0].spines['top'].set_linewidth(1.5)
axs[3,0].spines['right'].set_linewidth(1.5)
axs[3,0].spines['bottom'].set_linewidth(1.5)
axs[3,0].spines['left'].set_linewidth(1.5)
fig.colorbar(pcm20,ax = axs[3,0],extend="neither",location='left')#,shrink=0.80)
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

# here I've divided the DOS for the number of the respective atoms in the Supercell. 
#dosFe/27,enFe
axs[3,1].plot(dosAl/27,enAl-EfermiAl,linewidth = 1, color="k", label="Total") #marker=".", markersize=4, 
axs[3,1].plot(PdedosAl/27,ePdAl-EfermiAl,linewidth = 1, color="darkgoldenrod", label="Pd")
axs[3,1].plot(CoedosAl/26,eCoAl-EfermiAl,linewidth = 1, color="royalblue",label= "Co")
axs[3,1].plot(OedosAl/54,eOAl-EfermiAl,linewidth = 1, color="r", label = "O")
axs[3,1].plot(FeedosAl,eFeAl-EfermiAl,linewidth = 1, color="aqua", label = "Al")
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
#axs[2,1].set_xlabel('DOS(states/(eV)',fontsize=8,fontname='Ubuntu')
#axs[2,1].xaxis.set_label_coords(1.35, -0.)
axs[3,1].axhline(0, linewidth=1, color='k', alpha=0.8, linestyle = '-')
# bbox_to_anchor is basically to move the legends around--adjust it accordingly. 
axs[3,1].legend(loc='upper right',fontsize="8",prop=font,frameon=False,facecolor='none',borderpad=0.2,labelspacing=0.3,handlelength=1,handletextpad=0.4,borderaxespad=0.2,bbox_to_anchor=(1,0.6))




#########################################################################################33


pcm30 = axs[1, 0].pcolormesh(k1Ag,E1Ag,densityAg.T, cmap='magma', shading='auto', 
                             norm=colors.LogNorm(vmin= np.min(densityAg.T)+0.01, vmax= np.max(densityAg.T)))
axs[1,0].spines['top'].set_linewidth(1.5)
axs[1,0].spines['right'].set_linewidth(1.5)
axs[1,0].spines['bottom'].set_linewidth(1.5)
axs[1,0].spines['left'].set_linewidth(1.5)
fig.colorbar(pcm30,ax = axs[1,0],extend="neither",location='left')#,shrink=0.80)
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

# here I've divided the DOS for the number of the respective atoms in the Supercell. 
#dosFe/27,enFe
axs[1,1].plot(dosAg/27,enAg-EfermiAg,linewidth = 1, color="k", label="Total") #marker=".", markersize=4, 
axs[1,1].plot(PdedosAg/26,ePdAg-EfermiAg,linewidth = 1, color="darkgoldenrod", label="Pd")
axs[1,1].plot(CoedosAg/27,eCoAg-EfermiAg,linewidth = 1, color="royalblue",label= "Co")
axs[1,1].plot(OedosAg/54,eOAg-EfermiAg,linewidth = 1, color="r", label = "O")
axs[1,1].plot(FeedosAg,eFeAg-EfermiAg,linewidth = 1, color="aqua", label = "Ag")
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
#axs[2,1].set_xlabel('DOS(states/(eV)',fontsize=8,fontname='Ubuntu')
#axs[2,1].xaxis.set_label_coords(1.35, -0.)
axs[1,1].axhline(0, linewidth=1, color='k', alpha=0.8, linestyle = '-')
# bbox_to_anchor is basically to move the legends around--adjust it accordingly. 
axs[1,1].legend(loc='upper right',fontsize="8",prop=font,frameon=False,facecolor='none',borderpad=0.2,labelspacing=0.3,handlelength=1,handletextpad=0.4,borderaxespad=0.2,bbox_to_anchor=(1,0.6))




###########################################################################################

# to adjust the spacing between the three plots both as width and height. 
plt.subplots_adjust(wspace=0.04, hspace=0.15)

plt.savefig("Figure_DOS_w_Ag.png", dpi = 1200) #change to dpi 1200 for publishing.
