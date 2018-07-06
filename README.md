# Convolutional Neural Networks for the identification of signatures of natural selection and functional mutations in the human genome

# realdata_FASTA.py

REQUIREMENTS: 
Python 3.6

It is the script for real data images; it produces three multi-population images and three single-population images for each population contained in the file .FASTA. 

INPUT:
gene -- gene name; example: gene = MCM6
working_path -- example: "\Users\lucrezialorenzon\RESULT_1"

Create a folder in working_path named gene+"_images"; example: MCM6_images
All the images produced will be saved there as .bmp

apply_threshold_global -- boolean
delete_d -- boolean
threshold_global -- global filtering threshold. If delete_d = True the positions with derived allele frequency <= threshold_global will be eliminated. If delete_d = False the positions with minor allele frequency <= threshold_global will be eliminated.

apply_threshold -- boolean
p_threshold -- single population filtering threshold. If apply_threshold = True the position with ancestral allele frequency <= p_threshold will be eliminated. ONLY in black/white images with derived/ancestral alleles.

FASTA files
gene + "_anc.fasta" -- reference genome; example: MCM6_anc.fasta
gene + "_gene.fasta" -- populations genomes; example: MCM6_gene.fasta

# MSMS_images.py






