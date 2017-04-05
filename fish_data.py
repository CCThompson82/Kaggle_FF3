"""fish_data module contains the helper functions for the model build of the
Nature Conservancy Fisheries Kaggle Competition.

Dependencies:
    * numpy as np
    * os
    * scipy.ndimage as ndimage
    * scipy.misc as misc
    * scipy.special as special
    * matplotlib.pyplot as plt
    * tensorflow as tf
    * pickle

"""

#dependencies
import numpy as np
import os
from scipy import ndimage, misc, special
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

def generate_filenames_list(subdirectory = 'data/train/', subfolders = True) :
    """Returns a list of filenames in a given directory.  If subfolders is
    set to True, then fn will also iterate through all subfolders."""
    if subfolders :
        for i, species_ID in enumerate(os.listdir(subdirectory)[1:]) :
            fish_file_names = []
            fish_file_names = [subdirectory+species_ID+'/'+x for x in os.listdir(subdirectory+'/'+species_ID) ]
            fish_count = len(fish_file_names)

            try :
                master_file_names = master_file_names + fish_file_names
            except :
                master_file_names = fish_file_names
    else :
        master_file_names = [subdirectory+x for x in os.listdir(subdirectory)]
    return master_file_names




def show_panel(image) :
    """Shows an RGB montage of an image in array form."""
    plt.figure(figsize=(16,8))
    plt.subplot(1,4,1)
    plt.imshow(image[:,:,0], cmap = 'Reds')
    plt.subplot(1,4,2)
    plt.imshow(image[:,:,1], cmap = 'Greens')
    plt.subplot(1,4,3)
    plt.imshow(image[:,:,2], cmap = 'Blues')
    plt.subplot(1,4,4)
    plt.imshow(image)
    plt.show()

def boxit(f, coarse_dims = [64, 112, 3], fov_dim = 72) :
    """
    Function to efficiently annotate squared bbox from coarse images of
    a standard dimension (which may not necessarily reflect the original
    dimensions of the high-resolution image).
    """
    img = misc.imread(f, mode = 'RGB')
    shape = img.shape

    imgC = misc.imresize(img, size = coarse_dims, mode = 'RGB')

    show_panel(imgC)

    top_left_y = input("y coordinate of top left border   ")
    top_left_x = input("x coordinate of the top left border   ")
    bottom_right_y = input("y coordinate of bottom right border   ")
    bottom_right_x = input("x coordinate of bottom right border   ")

    tl = np.array([int(top_left_y), int(top_left_x)]) / np.array(coarse_dims[0:2])
    br = np.array([int(bottom_right_y), int(bottom_right_x)]) / np.array(coarse_dims[0:2])

    TL = np.round(tl * shape[0:2]).astype(int)
    BR = np.round(br * shape[0:2]).astype(int)

    dims = BR - TL
    dim = np.max(dims)
    ext = (dim - np.min(dims)) // 2

    if dims[0] > dims[1] :
        TL[1] = TL[1] - ext
    else :
        TL[0] = TL[0] - ext

    fov = img[TL[0]:TL[0]+dim, TL[1]:TL[1]+dim, :]

    show_panel(fov)

    adjust = input("Adjustments needed? (y/n)    ")

    if adjust == 'n' :
        proceed = True
    else :
        proceed = False
    while proceed == False :
        ad_horizontal = int(input("adjust left or right? (neg is left)   "  ))
        ad_vertical = int(input("adjust up or down? (neg is up)     "))
        ad_zoom = float(input("percent zoom (1.0 = same size, >1 -> zoom out)?    "))

        TL[0] = TL[0] + ad_horizontal
        TL[1] = TL[1] + ad_vertical

        dim = np.round(dim * ad_zoom).astype(int)

        fov = img[TL[0]:TL[0]+dim, TL[1]:TL[1]+dim, :]
        show_panel(fov)

        adjust = input("Adjustments needed? (y/n)    ")

        if adjust == 'n' :
            proceed = True
        else :
            proceed = False

    scale = fov_dim / dim
    # convert pixel offsets of the top left coordinate to the proportion of the y and x dimensions of the original high-resolution image
    return TL / shape[0:2], scale


def retrieve_fovea(f, top_left_coords, scale, fov_dim = 72) :
    """
    Retrieves a standard sized fovea based on the coordinates of the top left
    of a bounding box, plus the scale that yields an appropriate zoom for a
    circumscribing box of known size.
    """
    img = misc.imread(f, mode = 'RGB')
    img_shape = img.shape
    sc_img = misc.imresize(img, size = scale, mode = 'RGB')
    sc_shape = sc_img.shape
    offsets = np.round( top_left_coords * sc_shape[0:2]   ).astype(int)

    new_img = sc_img[offsets[0]:offsets[0]+fov_dim,
                     offsets[1]:offsets[1]+fov_dim,
                     :]
    return new_img
