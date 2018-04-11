import sys
import os
import nibabel as nib
import numpy as np

# Let's check the size of the PAC2018 niftis
workdir = os.path.expanduser('~/pac2018_root/')
filename = os.path.join(workdir, 'PAC2018_0001.nii')
img = nib.load(filename)

print(img.shape)

print(img.get_data_dtype())

hdr = img.header

print(hdr.get_xyzt_units())

atlas_filename = os.path.join(workdir, 'vols/BN_Atlas_246_1mm.nii')
atlas_img = nib.load(atlas_filename)

print(atlas_img.shape)

print(atlas_img.get_data_dtype())

hdr = atlas_img.header

print(hdr.get_xyzt_units())


# We add our wrappers, scripts and example_data folders to the searchpath
sys.path.append('./wrappers')
sys.path.append('./data')
sys.path.append('./scripts')

# We define our file locations. If you are using your own data these are the locations that you will want to change
centroids_file = "./data/brainnetome.centroids.txt"
names_file = "./data/brainnetome.names.txt"
regionalmeasures_file ="./data/PAC2018Covariates_and_regional_GMD.csv"

# We choose where to output our corrmat file
corrmat_file = os.getcwd()+'/corrmat_file.txt'

import corrmat_from_regionalmeasures as cfrm
cfrm.corrmat_from_regionalmeasures(regionalmeasures_file, names_file, corrmat_file, names_308_style=False)

import visualisation_commands as vc
corrmat_picture=os.getcwd()+'/corrmat_picture'
vc.view_corr_mat(corrmat_file, corrmat_picture, cmap_name='gnuplot2')
from IPython.display import Image
Image("corrmat_picture.png")

# Once again, this is going to write some files, so we give it a location
network_analysis = os.getcwd()+'/network_analysis'
# This is going to take a couple of minutes
import network_analysis_from_corrmat as nafc
network_analysis = os.getcwd()+'/network_analysis'
nafc.network_analysis_from_corrmat(corrmat_file,
                                  names_file,
                                  centroids_file,
                                  network_analysis,
                                  cost=10,
                                  n_rand=100,
                                  names_308_style=False)

# We will be using the files we just used network_analysis_from _corrmat to create
NodalMeasures=network_analysis+'/NodalMeasures_corrmat_file_COST010.csv'
GlobalMeasures=network_analysis+'/GlobalMeasures_corrmat_file_COST010.csv'
RichClub=network_analysis+'/RICH_CLUB_corrmat_file_COST010.csv'
# We're going to save them in a folder called figures
figures_dir=os.getcwd()+'/figures'

import make_figures as mfg
mfg.network_summary_fig(corrmat_file, NodalMeasures, GlobalMeasures, RichClub, figures_dir)
Image("figures/NetworkSummary_LowRes.png")