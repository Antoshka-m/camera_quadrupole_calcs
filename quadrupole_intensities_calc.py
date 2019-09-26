"""
Functions for calculation intensity difference signal
with 4-quadrant method. All functions applied to the single grayscale image.
"""
import matplotlib.pyplot as plt
import numpy as np
from  skimage.transform import rotate
from skimage.filters import threshold_otsu
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

def rotate_image(frame, angle=0):
    """
    Rotates images if needed (for the cantilever,
    where oscillation expected only in one direction)
    
    Parameters
    -------
    frame : np-array
        grayscale image
    angle: float
        angle in deg used to rotate
        
    Returns
    -------
    angle : float
        angle in deg used for rotation
    finish: Boolean
        True if shouldn't rotate anymore
        (condition to break the loop)
    """
    print("Is this image fine for you? Chosen angle is ", angle, "degrees\n")
    plt.figure()
    plt.imshow(rotate(frame, angle), cmap=plt.cm.gray)
    plt.show()
    a = input("Choose other angle or leave blank if its ok\n")
    if a=="":
        finish=True
    else:
        angle=float(a)
        finish=False
        print("finished rotating to angle ", angle, "degrees.")
    return angle, finish


def threshold_centroid (image):
    """
    Using threshold, determine centroid cooordinates.
    Coordinates of the centeroid of label which has the biggest area
    (assumed to be whole light spot, not some artefact) are returned.
    
    Parameters
    -------
    image: np-array
        grayscale image
        
    Returns
    -------
    center : tuple
        tuple of center coordinates
    """
    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh)
    # remove artifacts connected to image border
    #cleared = clear_border(bw)
    cleared = bw
    # label image regions
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=image)
    centers=[]
    areas=[]
    for region in regionprops(label_image):
        # draw rectangle around segmented areas
        # minr, minc, maxr, maxc = region.bbox
        areas.append(region.area)
        center = region.centroid
        centers.append(center)
    i=areas.index(max(areas))
    return centers[i]


def calc_intensities(image, center_coord, Iz, Il, Isum):
    """
    Calculation of the intensity difference TP, LR, and SUM
    for the qudrants of the single image. Calculated intensities
    added to the Il, Iz, Isum lists and returned to be used over
    during the loop.
    
    Parameters
    -------
    image: np-array
        grayscale image
    center_coord: tuple
        coordinates of the center of quadrants
    Iz: list
        vertical intensity difference
    Il: list
        horizontal intensity difference
    Isum: list
        sum of intensities
        
    Returns
    -------
    Iz: list
        vertical intensity difference
    Il: list
        horizontal intensity difference
    Isum: list
        sum of intensities
    """
    I_C = np.sum(image[0:int(round(center_coord[0], 0)), 0:int(round(center_coord[1], 0))])
    I_D = np.sum(image[0:int(round(center_coord[0], 0)), int(round(center_coord[1], 0)):])
    I_B = np.sum(image[int(round(center_coord[0], 0)):, int(round(center_coord[1], 0)):])
    I_A = np.sum(image[int(round(center_coord[0], 0)):, 0:int(round(center_coord[1], 0))])
    Iz=np.append(Iz, (I_A+I_B)-(I_C+I_D))
    Il=np.append(Il, (I_A+I_C)-(I_B+I_D))
    Isum=np.append(Isum, np.sum(image))
    return Iz, Il, Isum
