# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os 
import numpy as np


selection_coefficients = [0 ,  0.00025,  0.0005 ,  0.00075,  0.001  ,  0.00125,
        0.0015 ,  0.00175,  0.002  ,  0.00225,  0.0025 ,  0.00275,
        0.003  ,  0.00325,  0.0035 ,  0.00375,  0.004  ,  0.00425,
        0.0045 ,  0.00475,  0.005  ,  0.00525,  0.0055 ,  0.00575,
        0.006  ,  0.00625,  0.0065 ,  0.00675,  0.007  ,  0.00725,
        0.0075 ,  0.00775,  0.008  ,  0.00825,  0.0085 ,  0.00875,
        0.009  ,  0.00925,  0.0095 ,  0.00975,  0.01]

NCHROMS = 198
NITER = 10000
NPOS = 50000
NREF = 10000

gentime = 29 ## Tremblay and Vezina 2000
mu = 1.45e-8 
recrate = 1.0e-8 
theta = 4*NREF*mu*NPOS
rho = 4*NREF*recrate*NPOS

allelfreq = 0.73

os.chdir("/home/lucrezialorenzon/Software/msms")
for s in selection_coefficients:
    selclass = s 
    selstrength = s*2*NREF
    homoselstrength = selstrength*2
    comand = "java -jar ~/Software/msms/lib/msms.jar -N "+str(NREF)+" -ms "+str(NCHROMS)+" "+str(NITER)+" -t "+str(theta)+" -r "+str(rho)+" "+str(NPOS)+" -Sp 0.5 -SF 0 "+str(allelfreq)+" -SAA "+str(homoselstrength)+" -SAa "+str(selstrength)+" > /home/lucrezialorenzon/RESULT_5/Results/msms.."+str(s)+".txt"
    os.system(comand)



