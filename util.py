import numpy as np
import scipy
import glob
import os
import re

from PIL import Image

from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti

def merge(images, size, channels=3):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], channels))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def imsave(images, size, path):
    if images.shape[-1] > 4:
        for channel in xrange(4):
            new_path = path[0:-4] + '_' + str(channel) + '.png'
            scipy.misc.imsave(new_path, merge(images[..., channel][..., np.newaxis], size))
        return scipy.misc.imsave(path, merge(images[..., 4:], size))
    elif images.shape[-1] == 3:
        return scipy.misc.imsave(path, merge(images, size))
    elif images.shape[-1] == 1:
        scipy.misc.imsave(path, np.squeeze(merge(images[:,:,:,0][:,:,:,np.newaxis], size, channels=1)))
    else:
        scipy.misc.imsave(path, merge(images[:,:,:,:3], size))
        new_path = path[0:-4] + '_mask.png'
        return scipy.misc.imsave(new_path, np.squeeze(merge(images[:,:,:,3][:,:,:,np.newaxis], size, channels=1)))

def inverse_transform(image):
    return ((image + 1.)* 127.5).astype(np.uint8)

def add_parameter(class_object, kwargs, parameter, default=None):
    if parameter in kwargs:
        setattr(class_object, parameter, kwargs.get(parameter))
    else:
        setattr(class_object, parameter, default)

def save_images(images, size, image_path):
    data = inverse_transform(images)
    print data.shape
    return imsave(data, size, image_path)

def save_image(data, image_path):
    if image_path.endswith('png'):
        if data.shape[-1] == 4:
            data = data[:,:,:-1]
        return scipy.misc.imsave(image_path, data)
    # elif:
    #     image_path.endswith('nii.gz'):
    #     data = inverse_transform(image[0,...])
    #     for dim in xrange(data.shape[-1]):
    #         pass
    #     return
    # else:
    #     return

def try_int(s):
    "Convert to integer if possible."
    try: return int(s)
    except: return s

def natsort_key(s):
    "Used internally to get a tuple by which s is sorted."
    import re
    return map(try_int, re.findall(r'(\d+|\D+)', s))

def natcmp(a, b):
    "Natural string comparison, case sensitive."
    return cmp(natsort_key(a), natsort_key(b))

def natcasecmp(a, b):
    "Natural string comparison, ignores case."
    return natcmp(a.lower(), b.lower())

def stack_to_nii(image_directory, output_path='synthetic_brain'):

    images = glob.glob(os.path.join(image_directory, '*.png'))
    images.sort(natcasecmp)
    print images

    image_list = []
    for image in images:
        image_list += [np.asarray(Image.open(image), dtype=float)]
    image_stack = np.stack(image_list, axis=2)

    print image_stack.shape

    for dim in xrange(image_stack.shape[-1]):
        save_numpy_2_nifti(image_stack[...,dim], np.eye(4), output_path + '_' + str(dim) + '.nii.gz')

if __name__ == '__main__':

    stack_to_nii('./mri_slerp')

    pass