import os
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from joblib import Parallel, delayed

img_dir = '~/pac2018_root/'

atlas_name = 'brainnetome'

if atlas_name == 'brainnetome':
    atlas_filename = os.path.join(img_dir, 'vols/BN_Atlas_246_1mm.nii')
else:
    atlas_filename = os.path.join(img_dir,
                                  'vols/rcorrected_rwOASIS-TRT-20_DKT31_CMA_jointfusion_labels_in_MNI152_onlyGM.nii')

generate_GMD_files = False


def process_file(file):
    atlas_img = nib.load(atlas_filename)
    atlas_img_data = atlas_img.get_data()
    regions = np.unique(atlas_img_data[~np.isnan(atlas_img_data)])

    pattern = "PAC2018_"
    extension_filter = ['nii']
    if os.path.isfile(os.path.join(img_dir, file)):
        if file[-3:] in extension_filter and pattern in file:
            print("Processing %s" % file)
            img_filename = os.path.join(img_dir, file)
            filename_no_ext = file[:-4]
            dest_filename = os.path.join(img_dir, filename_no_ext + '_GM-' + atlas_name + '.txt')

            if not os.path.exists(dest_filename):
                img = nib.load(img_filename)
                resampled_img = resample_to_img(img, atlas_img)
                resampled_img_data = resampled_img.get_data()

                gmd = np.zeros((len(regions), 2))
                gmd[:, 0] = regions
                for region in regions:
                    r = int(region)
                    idx = np.where(atlas_img_data == r)
                    local_gmd = resampled_img_data[idx]
                    gmd[r, 1] = np.mean(local_gmd)

                np.savetxt(dest_filename, gmd)


if __name__ == '__main__':
    if generate_GMD_files:
        Parallel(n_jobs=4)(delayed(process_file)(f) for f in os.listdir(img_dir))

    # Let's check the size of the PAC2018 niftis
    filename = os.path.join('C:/Users/joshu/Desktop/PAC2018/', 'PAC2018_0001.nii')
    img = nib.load(filename)

    print(img.shape)

    print(img.get_data_dtype())

    hdr = img.header

    print(hdr.get_xyzt_units())

    atlas_img = nib.load(atlas_filename)

    print(atlas_img.shape)

    print(atlas_img.get_data_dtype())

    hdr = atlas_img.header

    print(hdr.get_xyzt_units())
