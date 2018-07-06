# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from Bio import SeqIO
from PIL import Image
import numpy as np
import copy

#n_gen = numero di genomi
#n_positions = numero di posizioni dell'allineamento
#matrix è la matrice delle posizioni alleliche. Dimensioni: n_gen x n_positions x 4. 
#freq è la matrice delle frequenze alleliche. Dimensioni: 4 x n_positions
#indici: 0-A, 1-C, 2-G, 3-T.
#n_alleles è un vettore contenente il numero di alleli presenti in ogni sito. Dimensioni: 1 x n_positions
def start (ref,records,alleles,p_order,populations,threshold,delete_d,delete_mono,delete_bi,delete_tri,delete_quadri,apply_threshold_global):
    #ESTRAGGO TUTTE LE POPOLAZIONI DAL FILE
    if populations == []:
       pops = []
       for i in range(len(records)):
           pops.append(records[i].name)
           populations = np.unique(pops)
    #n --> numero di popolazioni nel file/popolazioni specificate dall'utente
    n = len(populations)
    #ORDINO LE POPOLAZIONI
    populations_ordered = [0]*n
    index = 0
    for population in (p_order):
        if (population in populations):
            populations_ordered[index] = population
            index += 1
    #populations --> contiene popolazioni ordinate
    populations = populations_ordered
    #LEGGO I GENOMI DI OGNI POPOLAZIONE
    genomes = []
    #tot_n_gen --> vettore che contiene il numero di individui di ogni popolazione
    #(ordine dato da populations)
    tot_n_gen = [0]*n  
    for p,population in enumerate(populations):
        #p_n_gen --> contatore per individui di ogni popolazione
        p_n_gen = 0
        for i in range(len(records)):
            if records[i].name == population:
               p_n_gen += 1
               genome = []   
               for j in range(len(records[i].seq)):
                   genome.append(records[i].seq[j])
               genomes.append(genome)
        tot_n_gen[p] = p_n_gen
    #total_genomes --> contiene i genomi di tutte le popolazioni, una dopo l'altra
    total_genomes = np.array(genomes)
    #LEGGO IL GENOMA DI RIFERIMENTO
    ref_genome = []
    for i in range (len(ref.seq)):
        ref_genome.append (ref.seq[i])
    ref_genome = np.array(ref_genome)
    #creo la matrice delle posizioni alleliche matrix: 
    #il primo strato nella terza dimensione si riferisce alla A ed ha 1 dove c'e' una A
    #n_gen --> numero totale di individui (tutte le popolazioni insieme)
    n_gen = total_genomes.shape[0]
    n_positions = total_genomes.shape[1]
    matrix = np.zeros((n_gen,n_positions,4))  
    for i,letter in enumerate(alleles):
        xy = np.where(total_genomes == letter)
        xyz = list(xy)
        z = np.ones((len(xy[0])),dtype=int)*i
        xyz.append(z)
        matrix[xyz] = 1
    #Calcolo le frequenze alleliche
    freq, n_alleles = frequencies (alleles,n_gen,n_positions,matrix)
    #ELIMINO posizioni mono/bi/tri/quadri -alleliche
    arguments = [delete_mono,delete_bi,delete_tri,delete_quadri]
    if delete_mono==delete_bi==delete_tri==delete_quadri==False:
       pass
    else:
          for i,arg in enumerate(arguments): 
              if arg == True:
                 positions = np.where(n_alleles==i+1)
                 mask,n_positions,matrix,freq,n_alleles,ref_genome = delete (n_positions,matrix,freq,n_alleles,ref_genome,positions)
    #CALCOLO MINOR E MAIOR
    Mm = np.zeros((2,n_positions), dtype=np.str)
    freq1 = copy.copy(freq)
    for k in range(n_positions):
        max1 = freq[:,k].argmax(axis=0)
        freq1[max1,k] = 0
        s = freq1[:,k].sum(axis=0)
        if s == 0:
           max2 = max1
        else:
           max2 = freq1[:,k].argmax(axis=0)
        for i,letter in enumerate(alleles):
            if max1 == i:
               Mm [0,k] = letter
            if max2 == i:
               Mm [1,k] = letter
   #filtraggio per frequenza 
   #se delete_d = True eliminiamo le posizioni in cui la frequenza dell'allele derivato è minore o uguale di threshold
   #altrimenti eliminiamo le posizioni in cui la frequenza dell'allele minor è minore o uguale di threshold
   if apply_threshold_global = True:
      f_del = np.zeros(n_positions)
      if delete_d == True:
         for k in range(n_positions):
             for i,letter in enumerate(alleles):
                 if ref_genome[k] == letter:
                    f_del[k]=1-(freq[i,k])
      else:
          for k in range(n_positions):
             for i,letter in enumerate(alleles):
                 if Mm[1,k] == letter:
                    f_del[k]=freq[i,k]
      positions = np.where(f_del<=threshold)
      mask,n_positions,matrix,freq,n_alleles,ref_genome = delete (n_positions,matrix,freq,n_alleles,ref_genome,positions)
      Mm = Mm[:,mask]
      f_del = f_del[mask]
    return [populations,matrix,tot_n_gen,ref_genome,freq,Mm,n_positions,n,n_alleles]

def delete (n_positions,matrix,freq,n_alleles,ref_genome,positions):
    mask = np.ones(n_positions, dtype=bool)
    mask[positions] = False
    matrix = matrix [:,mask,:]
    n_positions = matrix.shape[1]
    freq = freq [:,mask]
    n_alleles = n_alleles [mask]
    ref_genome = ref_genome[mask]
    return [mask,n_positions,matrix,freq,n_alleles,ref_genome]

def frequencies (alleles,n_gen,n_positions,matrix):
    freq = np.zeros((4,n_positions))
    for i,letter in enumerate(alleles):
        freq[i,:] = matrix[:,:,i].sum(axis=0)/n_gen
        Vfreq = copy.copy(freq)
        Vfreq[Vfreq>0]=1
        n_alleles = Vfreq.sum(axis=0)
    return [freq,n_alleles]

def order (im_matrix):
    u,index,count = np.unique(im_matrix,return_index=True,return_counts=True,axis=0)
    b = np.zeros((np.size(im_matrix,0),np.size(im_matrix,1)),dtype=int)
    c = np.stack((index,count), axis=-1)
    c = c[c[:,1].argsort(kind='mergesort')]
    pointer = np.size(im_matrix,0)-1
    for j in range(np.size(c,0)):
        for conta in range(c[j,1]):
            b [pointer,:] = im_matrix[c[j,0]]
            pointer -= 1
    return b

def full_colour (n_gen,n_positions,matrix,population): 
    global c
    colours = [[100,0,0,0],[0,100,0,0],[0,0,100,0],[0,0,0,100]]
    colour_genomes = np.zeros((n_gen,n_positions,4))
    for i,colour in enumerate(colours):
        index = np.where(matrix[:,:,i]==1)
        colour_genomes[index] = colour
   #ORDINO LE RIGHE
   #u,index,count = np.unique(colour_genomes,return_index=True,return_counts=True,axis=0)
   #b = np.zeros((n_gen,n_positions,4),dtype=int)
   #c = np.stack((index,count), axis=-1)
   #c = c[c[:,1].argsort(kind='mergesort')]
   #pointer = n_gen-1
   #for j in range(np.size(c,0)):
   #    for conta in range(c[j,1]):
   #        b [pointer,:,:] = colour_genomes[c[j,0],:,:]
   #        pointer -= 1
   #colour_genomes = b
    colour_genomes_uint8 = np.uint8(colour_genomes)
    colour_genomes_im = Image.fromarray(colour_genomes_uint8, mode = 'CMYK').convert('RGB')
    string = workink_path + folder + "/img_fullcolor_" + population + ".bmp"
    colour_genomes_im.save (string)

#black_white crea l'immagine di tutti i genomi in bianco e nero, rispetto al
#al genoma di riferimento: bianco = allele ancestrale, nero = allele derivato.
# 1 [mode L = 255]--bianco, 0 [mode L = 0]--nero
def black_white (n_gen,n_positions,alleles,matrix,ref_genome,population,p_threshold,apply_threshold):
    bw_genomes = np.zeros((n_gen,n_positions))
    freq_a = np.zeros((1,n_positions))
    for k in range(n_positions):
        for i,letter in enumerate(alleles):
            if ref_genome[k] == letter:
               #bw_genomes ha 1 dove l'allele è quello ancestrale
               bw_genomes[:,k] = matrix[:,k,i]
               freq_a[0,k] = bw_genomes[:,k].sum(axis=0)/n_gen
    if apply_threshold == True:
       #elimino le posizioni in cui la frequenza dell'allele ancestrale è minore di p_threshold
       positions = np.where(freq_a <= p_threshold)
       mask = np.ones(n_positions, dtype=bool)
       mask[positions[1]] = False
       bw_genomes = bw_genomes [:,mask]
       #n_p_positions = bw_genomes.shape[1]
   #ORDINO LE RIGHE
   #bw_genomes = order(bw_genomes)
    bw_genomes_uint8 = np.uint8(bw_genomes)
    bw_genomes_im = Image.fromarray (bw_genomes_uint8*255, mode = 'L')
    string = workink_path + folder + "/img_bw_" + population + ".bmp"
    bw_genomes_im.save (string)
                
#black_white_Mm crea l'immagine di tutti i genomi in bianco e nero, rispetto agli alleli 
#a frequenza maggiore e minore: bianco = maior, nero = minor. 
# 1 [mode L = 255]--bianco, 0 [mode L = 0]--nero
#Mm è un vettore la cui prima riga contiene il maior allele di ogni posizione
#e la seconda riga contiene il minor allele di ogni posizione. Dimensioni: 2 x n_positions
def black_white_Mm (n_gen,n_positions,Mm,matrix,alleles,population):
    bw_Mm_genomes = np.zeros((n_gen,n_positions))
    for k in range(n_positions):
        for i,letter in enumerate(alleles):
            if Mm[0,k]==letter:
               #bw_Mm_genomes ha 1 dove l'allele e' quello maior
               bw_Mm_genomes[:,k]=matrix[:,k,i]
   #ORDINO LE RIGHE
   #bw_Mm_genomes = order(bw_Mm_genomes)
    bw_Mm_genomes_uint8 = np.uint8(bw_Mm_genomes)
    bw_Mm_genomes_im = Image.fromarray (bw_Mm_genomes_uint8*255, mode = 'L')
    string = workink_path + folder + "/img_bw_Mm_" + population + ".bmp"
    bw_Mm_genomes_im.save (string)

def colour_freq (freq,n_positions):
    freq = np.reshape(freq.T, (1,n_positions,4))
    colour_freq_uint8 = np.uint8(freq)
    return colour_freq_uint8  

def Mm_freq (Mm,n_positions,freq):
    m = np.zeros((1,n_positions))
    for k in range(n_positions):
        for i,letter in enumerate(alleles):
            if Mm[0,k]==letter:
               #m e' tanto piu'bianco quanto piu' e' alta la frequenza del maior
               #che e' come dire e' tanto piu' nero quanto piu' e' alta la freq. del minor
               m[0,k] = np.rint(freq[i,k]*2.55)
    m_uint8 = np.uint8(m)
    return m_uint8

def Ad_freq (n_gen,n_positions,alleles,matrix,ref_genome,population):
    bw_genomes = np.zeros((n_gen,n_positions))
    freq_a = np.zeros((1,n_positions))
    for k in range(n_positions):
        for i,letter in enumerate(alleles):
            if ref_genome[k] == letter:
               #bw_genomes ha 1 dove l'allele è quello ancestrale
               bw_genomes[:,k] = matrix[:,k,i]
               freq_a[0,k] = bw_genomes[:,k].sum(axis=0)/n_gen
    #freq_d e' tanto piu' nero quanto piu' e' alta la freq. dell'allele derivato
    freq_d = np.rint(freq_a*100)
    freq_d = np.rint(freq_d*2.55)
    freq_d_uint8 = np.uint8(freq_d)
    return freq_d_uint8

                                        #MAIN

p_order = ['LWK','ESN','YRI','MSL','GWD','ASW','ACB','TSI','IBS','CEU','GBR','FIN','ITU','STU','PJL','GIH','BEB','CHS','CHB','CDX','JPT','KHV','MXL','PUR','CLM','PEL']

working_path = "/Users/lucrezialorenzon/RESULT_1/"
gene = "MCM6"
folder = gene + "_images"
alleles = ['A','C','G','T']
#con populations = [] prendiamo tutte le popolazioni presenti nel file, altrimenti specifichiamo la lista delle pop. di interesse
populations = []
#filtraggio globale (apply_threshold_global) per frequenza 
#se delete_d = True eliminiamo le posizioni in cui la frequenza dell'allele derivato è minore o uguale di threshold
#altrimenti eliminiamo le posizioni in cui la frequenza dell'allele minor è minore o uguale di threshold
apply_threshold_global = False
threshold = 0.05
#p_threshold la mettiamo nelle immagini im_bw (funzione black_white); possiamo eliminare le posizioni in cui 
#la frequenza dell'allele ancestrale è minore o uguale di p_threshold
apply_threshold = False
p_threshold = 0.05
ref = SeqIO.read (workink_path + gene +"_anc.fasta","fasta")
records = list(SeqIO.parse(workink_path + gene +"_gene.fasta","fasta"))

populations,matrix,tot_n_gen,ref_genome,freq,Mm,n_positions,n,n_alleles = start (ref,records,alleles,p_order,populations,threshold,False,True,False,False,False,apply_threshold_global)

def images (populations,matrix,tot_n_gen,ref_genome,freq,Mm,n_positions,n,n_alleles,p_threshold,apply_threshold):                                                                             

    for p,population in enumerate(populations):
        from_gen = sum(tot_n_gen [:p])
        to_gen = from_gen + tot_n_gen[p]
        n_gen = (to_gen - from_gen) 
        p_matrix = matrix[from_gen:to_gen,:,:]
        p_freq,p_n_alleles = frequencies (alleles,n_gen,n_positions,p_matrix)
        p_freq = np.rint(p_freq*100)
        if p==0:
           poli_freq_uint8 = colour_freq (p_freq,n_positions)
           poli_m_uint8 = Mm_freq (Mm,n_positions,p_freq)
           poli_d_uint8 = Ad_freq (n_gen,n_positions,alleles,p_matrix,ref_genome,population)
        elif p == n-1:
             colour_freq_uint8 = colour_freq (p_freq,n_positions)
             poli_freq_uint8 = np.concatenate((poli_freq_uint8, colour_freq_uint8),axis=0)
             poli_freq_im = Image.fromarray(poli_freq_uint8, mode = 'CMYK').convert('RGB')
             string = workink_path + folder + "/poli_freq.bmp"
             poli_freq_im.save (string)
     
             m_uint8 = Mm_freq (Mm,n_positions,p_freq)
             poli_m_uint8 = np.concatenate((poli_m_uint8, m_uint8),axis=0)
             poli_m_im = Image.fromarray(poli_m_uint8, mode = 'L')
             string = workink_path + folder + "/poli_m.bmp"
             poli_m_im.save (string)
       
             freq_d_uint8 = Ad_freq (n_gen,n_positions,alleles,p_matrix,ref_genome,population)
             poli_d_uint8 = np.concatenate((poli_d_uint8, freq_d_uint8),axis=0)
             poli_d_im = Image.fromarray(poli_d_uint8, mode = 'L')
             string = workink_path + folder + "/poli_d.bmp"
             poli_d_im.save (string)
        else:
             colour_freq_uint8 = colour_freq (p_freq,n_positions)
             poli_freq_uint8 = np.concatenate((poli_freq_uint8, colour_freq_uint8),axis=0)
             m_uint8 = Mm_freq (Mm,n_positions,p_freq)
             poli_m_uint8 = np.concatenate((poli_m_uint8, m_uint8),axis=0)
             freq_d_uint8 = Ad_freq (n_gen,n_positions,alleles,p_matrix,ref_genome,population)
             poli_d_uint8 = np.concatenate((poli_d_uint8, freq_d_uint8),axis=0)
    
        full_colour (n_gen,n_positions,p_matrix,population)
        black_white_Mm(n_gen,n_positions,Mm,p_matrix,alleles,population)
        black_white (n_gen,n_positions,alleles,p_matrix,ref_genome,population,p_threshold,apply_threshold)
 
images (populations,matrix,tot_n_gen,ref_genome,freq,Mm,n_positions,n,n_alleles,p_threshold,apply_threshold)
    
