import os
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from joblib import Parallel, delayed

from make_dataset import count_pac2018

img_dir = os.path.expanduser('~/pac2018_root/')

atlas_name = 'brainnetome'

if atlas_name == 'brainnetome':
    atlas_filename = os.path.join(img_dir, 'vols/BN_Atlas_246_1mm.nii')
else:
    atlas_filename = os.path.join(img_dir,
                                  'vols/rcorrected_rwOASIS-TRT-20_DKT31_CMA_jointfusion_labels_in_MNI152_onlyGM.nii')

generate_GMD_files = True


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
            dest_filename_mean = os.path.join(img_dir, filename_no_ext + '_GM-mean-' + atlas_name + '.txt')
            dest_filename_var = os.path.join(img_dir, filename_no_ext + '_GM-var-' + atlas_name + '.txt')

            if not os.path.exists(dest_filename_mean):
                img = nib.load(img_filename)
                resampled_img = resample_to_img(img, atlas_img)
                resampled_img_data = resampled_img.get_data()

                gmd_mean = np.zeros((len(regions), 2))
                gmd_var = np.zeros((len(regions), 2))
                gmd_mean[:, 0] = regions
                gmd_var[:, 0] = regions
                for region in regions:
                    r = int(region)
                    idx = np.where(atlas_img_data == r)
                    local_gmd = resampled_img_data[idx]
                    gmd_mean[r, 1] = np.mean(local_gmd)
                    gmd_var[r, 1] = np.var(local_gmd)

                np.savetxt(dest_filename_mean, gmd_mean)
                np.savetxt(dest_filename_var, gmd_var)


if __name__ == '__main__':
    if generate_GMD_files:
        #Parallel(n_jobs=4)(delayed(process_file)(f) for f in os.listdir(img_dir))

        n = count_pac2018(img_dir, 'PAC2018Covariates_and_regional_GMD.csv')

        atlas_img = nib.load(atlas_filename)
        atlas_img_data = atlas_img.get_data()
        regions = np.unique(atlas_img_data[~np.isnan(atlas_img_data)])

        mean_pattern = "_GM-mean-"
        var_pattern = "_GM-var-"
        gmd_means = np.dtype({'names': ('subject', 'region', 'mean'), 'formats': (int, int, np.float32)})
        gmd_vars = np.dtype({'names': ('subject', 'region', 'var'), 'formats': (int, int, np.float32)})
        extension_filter = ['txt']
        for file in os.listdir(img_dir):
            if os.path.isfile(os.path.join(img_dir, file)) and file[-3:] in extension_filter:
                filename = file[:12]
                number = int(file[8:12])
                if mean_pattern in file:
                    print("%s has mean data" % filename)
                    gm_mean = np.loadtxt(os.path.join(img_dir, file))
                    gm_mean = np.append(np.repeat(number, len(gm_mean)), gm_mean)
                    gmd_means = np.append(gmd_means, gm_mean)
                if var_pattern in file:
                    print("%s has variance data" % filename)
                    gm_var = np.loadtxt(os.path.join(img_dir, file))
                    nums = np.repeat(number, len(gm_var))
                    print(nums.shape)
                    gm_var = np.append(nums, gm_var)

                    gmd_vars = np.append(gmd_vars, gm_var)

    print(gmd_means.shape)
    print(gmd_vars.shape)

    # Let's check the size of the PAC2018 niftis
    filename = os.path.join(img_dir, 'PAC2018_0001.nii')
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
