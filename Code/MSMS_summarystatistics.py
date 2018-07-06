#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:58:38 2018

@author: lucrezialorenzon
"""
import numpy as np
import allel
import csv

def statistics (S, N, file_name, NCHROMS):
    global once
    global nsl
    #importo il file
    file = open("/home/lucrezialorenzon/Simulations/Results_decompressed/" + file_name).readlines()
    #cerco il carattere // nel file 
    find = []
    for i, string in enumerate(file):
        if string == '//\n':
           find.append(i+3)
    for ITER,pointer in enumerate(find):
    #croms è la matrice totale
        for j in range(NCHROMS):
            f = list(file[pointer + j])
            del f[-1]
            pos = file[pointer - 1].split()
            del pos[0]
            pos = np.array(pos, dtype='float')
            pos = pos*100000 #perchè abbiamo simulato una regione di 100000 posizioni 
            F = np.array(f,dtype=int)
            if j == 0:
               croms = F
            else:
               croms = np.vstack((croms,F))
        #n_pos è il numero di posizioni
        n_pos = np.size(croms,1)
        freq = croms.sum(axis=0)/NCHROMS
        freq = np.array(freq)
    
        positions_1 = np.where(freq<0.70)
        mask_1 = np.ones(n_pos, dtype=bool)
        mask_1[positions_1[0]] = False
        freq = freq[mask_1]
        n_pos_1 = np.size(freq)
        positions_2 = np.where(freq>0.90)
        mask_2 = np.ones(n_pos_1, dtype=bool)
        mask_2[positions_2[0]] = False
        freq = freq[mask_2]
        
        #SUMMARY STATISTICS 
        haplos = np.transpose(croms)
        h = allel.HaplotypeArray(haplos)
        #tajimasd
        ac = h.count_alleles()
        TjD = allel.stats.tajima_d(ac)
        #watterson
        theta_hat_w = allel.stats.watterson_theta(pos, ac)
        #nsl
        nsl = allel.nsl(h)
        nsl = nsl[mask_1]
        nsl = nsl[mask_2]
        size = np.size(nsl)
        if size == 0:
            nsl_max = 0
        else:
            nsl_max = np.max(nsl)
        #scrivo su file csv
        f = open("/home/lucrezialorenzon/Simulations/summarystatistics.csv",'a+')
        with f:
            header = ['Selection coefficient','Population size','Iteration','Tajimas D','Watterson','nsl']
            writer = csv.DictWriter(f,fieldnames=header)
            if once == 0:
                writer.writeheader()
                writer.writerow({'Selection coefficient':str(S),'Population size':str(N),'Iteration':str(ITER+1),
                                 'Tajimas D':TjD,'Watterson':theta_hat_w,'nsl':nsl_max})
                once = 1
            else: 
                writer.writerow({'Selection coefficient':str(S),'Population size':str(N),'Iteration':str(ITER+1),
                                 'Tajimas D':TjD,'Watterson':theta_hat_w,'nsl':nsl_max})
         
#%%      
                                        #MAIN
                                        
selection_coefficients = [0,0.0025,0.005,0.0075,0.01]
NREF = [5000,7500,10000,20000,50000]  
NCHROMS = 100
NITER = 10000

global once
once = 0

for S in selection_coefficients:
    for N in NREF:
        file_name = "msms.."+ str(S) + ".." + str(N) + ".txt"
        statistics (S, N, file_name, NCHROMS)

#%%       
        
        
        
        
        