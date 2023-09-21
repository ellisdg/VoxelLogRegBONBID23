import ants
import glob
import torch
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk
import tqdm
import os
import shutil
from multiprocessing import Pool
from functools import partial

ground_truth_filenames = sorted(glob.glob("./BONBID2023_Train/3LABEL/*.mha"))
template_filename = "./template10.nii.gz"

# Now register all of the images to the adc image selected

def new_image_like(array, image):
    return ants.from_numpy(array, 
                           origin=ants.get_origin(image), 
                           spacing=ants.get_spacing(image),
                           direction=ants.get_direction(image))

def warp_to_template(label_filename, tmp_fn, iter_dir, method):
    adc_filename = label_filename.replace("_lesion.mha", "-ADC_ss.mha",).replace("3LABEL", "1ADC_ss")
    if tmp_fn == adc_filename:
        return adc_filename, label_filename
    template_image = ants.image_read(tmp_fn)
    adc_image = ants.image_read(adc_filename)
    label_image = ants.image_read(label_filename)
    sub = os.path.basename(label_filename).split(".")[0]
    out_dir = os.path.abspath(os.path.join(iter_dir, sub))
    os.makedirs(out_dir, exist_ok=True)
    adc_out = os.path.join(out_dir, "adc.nii.gz")
    label_out = os.path.join(out_dir, "label.nii.gz")
    if os.path.exists(adc_out) and os.path.exists(label_out):
        return adc_out, label_out
    print(f"{adc_filename} --> {tmp_fn}")
    reg = ants.registration(fixed=template_image, moving=adc_image, type_of_transform=method,
                            outprefix=out_dir + "/")
    
    reg_adc_image = ants.apply_transforms(fixed=template_image, moving=adc_image,
                                          transformlist=reg['fwdtransforms'], 
                                          interpolator="lanczosWindowedSinc")
    
    reg_label_image = ants.apply_transforms(fixed=template_image, moving=label_image,
                                            transformlist=reg['fwdtransforms'], 
                                            interpolator="genericLabel")
    

    print(f"Writing {adc_out}")
    ants.image_write(reg_adc_image, adc_out)
    ants.image_write(reg_label_image, label_out)
    del reg, reg_adc_image, reg_label_image, adc_image, label_image
    return adc_out, label_out

n_threads = 20
for i in (11,):

    iter_dir = "./antsoutputs/final"
    if i == 0:
        method = "Affine"
    elif i < 6:
        method = "antsRegistrationSyNQuick"
    else:
        method = "antsRegistrationSyN"
    
    warp_func = partial(warp_to_template, tmp_fn=template_filename, iter_dir=iter_dir, method=method)
    print("starting registration pool")
    with Pool(n_threads) as pool:
        outputs = pool.map(warp_func, ground_truth_filenames)
    
    print("completed registration, compiling template")
    
    
    _template = np.stack([np.asarray(ants.image_read(outputs[i][0]).numpy(), float) for i in range(len(outputs))])
    _labels = np.stack([ants.image_read(outputs[i][1]).numpy() for i in range(len(outputs))])
    np.save("labels.npy", _labels)
    np.save("template.npy", _template)
    # only use the voxels that are not in the label
    _template[_labels > 0.5] = 0
    _template = _template.sum(axis=0)

    _template = _template/((_labels < 0.5).sum(axis=0))
    
    print("writing template to file")
    template_image = new_image_like(_template, ants.image_read(template_filename))
    template_filename = "template_final.nii.gz".format(i)
    ants.image_write(template_image, template_filename)
        