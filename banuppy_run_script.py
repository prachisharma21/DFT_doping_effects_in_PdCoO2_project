#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 12:30:42 2022

@author: prachi
"""

import banduppy # Need to install banduppy package 

import shutil,os,glob
from subprocess import run
import numpy as np
import pickle

#QEpath=""

nproc=128
npkk = 1
PWSCF="mpirun -np {np} pw.x -nk {npk} ".format(np=nproc,npk=npkk).split()


unfold_path=banduppy.UnfoldingPath(
            supercell= [[3 ,  0 , 0],
                        [0 , 3 , 0],
                        [0  , 0 ,3]] ,   # How the SC latticevectors are expressed in the PC basis (should be a 3x3 array of integers)
            pathPBZ=[[0,0,0],[1/2,0,0],[1/2,1/2,0], [0,0,0],[1/2,1/2,1/2]],  # Path nodes in reduced coordinates in the primitive BZ. if the segmant is skipped, put a None between nodes
            nk=(70,70,90,26),  #  number of k-points in each non-skipped segment. 
            labels="GLFGT" )   # or ['L','G','X','U','K','G']



unfold=banduppy.Unfolding(
            supercell= [[3 ,  0 , 0],
                        [0 , 3 , 0],
                        [0  , 0 ,3]]  , # SC latticevectors are expressed in the PC basis (should be a 3x3 array of integers)
            kpointsPBZ =  np.array([np.linspace(0.0,0.5,12)]*3).T # just a list of k-points (G-L line in this example)
                              )

kpointsPBZ=unfold_path.kpoints_SBZ_str()   # a string  containing the k-points to be entered into the PWSCF input file  after 'K_POINTS crystal ' line. maybe transformed to formats of other codes  if needed

try:
    print ("unpickling unfold")
    unfold_path=pickle.load(open("unfold-path.pickle","rb"))
    unfold=pickle.load(open("unfold.pickle","rb"))
except Exception as err:
    print("error while unpickling unfold '{}',  unfolding it".format(err))
    try:
        print ("unpickling bandstructure")
        bands=pickle.load(open("bandstructure.pickle","rb"))  
        print ("unpickling - success")
    except Exception as err:
        print("Unable to unpickle  bandstructure '{}' \n  Reading bandstructurefrom .save folder ".format(err))
        try: 
            bands=banduppy.BandStructure(code="espresso", prefix="PdCoO2_Ag_defect_SC333")
        except Exception as err:
            print("error reading  bandstructure '{}' \n calculating it".format(err))
            pw_file="PdCoO2_Ag_SC333_"
            shutil.copy("./inputs/"+pw_file+"1.scf.in",".")
            open(pw_file+"2.scf.in","w").write(open("./inputs/"+pw_file+"2.scf.in").read()+kpointsPBZ)
            scf_run=run(PWSCF,stdin=open(pw_file+"1.scf.in"),stdout=open(pw_file+"1.scf.out","w"))
            bands_run=run(PWSCF,stdin=open(pw_file+"2.scf.in"),stdout=open(pw_file+"2.scf.out","w"))
            for f in glob.glob("*.wfc*"):
                os.remove(f)
            bands=banduppy.BandStructure(code="espresso", prefix="PdCoO2_Ag_defect_SC333")
        pickle.dump(bands,open("bandstructure.pickle","wb"))

    unfold_path.unfold(bands,break_thresh=0.1,suffix="path")
    unfold.unfold(bands,suffix="GL")
    pickle.dump(unfold_path,open("unfold-path.pickle","wb"))
    pickle.dump(unfold,open("unfold.pickle","wb"))

# use the data to plot in any other format
data=np.loadtxt("bandstructure_unfolded-path-morepts.txt")
