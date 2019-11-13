import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import skvideo.io
from skimage import io
from skimage.filters import threshold_mean  # simple mean thresholding
from skimage.segmentation import clear_border
from skimage.viewer import ImageViewer
from skimage.transform import AffineTransform, warp
# now import all needed functions from our own modules
from get_filenames import get_filenames
from calib_curve_im_and_line_plots import plot_im_w_quadrants, plot_shift_curves
from quadrupole_intensities_calc import rotate_image, threshold_centroid, calc_intensities
from calc_calib_line import calc_calib_line
from  skimage.transform import rotate


def main():
    # Let's import first frame of videofile, and show it
    print('Choose videofile for making calib curves')
    file = get_filenames()
    file = file[0] # get just single file instead of list
    print('Importing file ', file)
    frame = skvideo.io.vread(file, num_frames=1) # import just first frame
    frame = rgb2gray(frame[0]) # get element instead of list, make grayscale
    # plt.figure()
    # plt.imshow(frame, cmap=plt.cm.gray)
    # plt.show()
    
    # Compensate angle if its needed
    
    finish=False
    angle=0
    while finish == False:
            angle, finish=rotate_image(frame, angle)
    if angle != 0:
        frame=rotate(frame, angle)
    
    # Detect center of lightspot, show quadrants:
    centroid=threshold_centroid(frame)
    print('Showing first frame of video with quadrants...')
    plot_im_w_quadrants(frame, centroid, fig_title='1st frame with quadrants')
    
    
    # Demonstrate how shifted image looks like
    print('Image shifted for 5px in each axis will look like this:')
    transform = AffineTransform(translation=(5, 5))
    shifted = warp(frame, transform, mode='constant', preserve_range=True)
    plot_im_w_quadrants(shifted, centroid, fig_title='5 px shift')
    print('If you want to have max test shift as shown above, just press enter')
    print('(If lightspot is partially out of field of view, better to choose smaller shift)')
    print('Otherwise manually enter desired shift of image in px, and press enter')
    print('Note: the same shift will be used for each axis')
    max_shift = input()
    if max_shift == '':
        max_shift = 5
    else:
        max_shift = float(max_shift)
    print('Images will be shifted from 0 to %s px' %max_shift)
    k_px_um = input('Enter px to um coefficient (scale):\n')
    k_px_um = float(k_px_um)
    # Shift images along x axis
    
    shifted_im = []
    # specify parameters for calculations
    # generate dx value for linear shift
    # x_shift = np.array([0.1*dx for dx in range(0, max_shift+1)])
    dx=0.1
    x_shift = np.arange(0, max_shift*(1+dx), dx*max_shift)
    # k_px_um = 1.36 # scale px to um
    normalization = False # don't scale signal over SUM
    shift_vs_sig = True # calculate shift from signal, not other way
    for dx in x_shift:
        transform = AffineTransform(translation=(dx, dx))  # shift along both axis
        shifted_im.append(warp(frame,
                               transform,
                               mode='constant',
                               preserve_range=True))
    
    # Calculate the intensities
    
    Il=np.array([])
    Iz=np.array([])
    Isum=np.array([])
    for i in range(len(shifted_im)):
        Iz, Il, Isum = calc_intensities(shifted_im[i],
                                        centroid,
                                        Iz,
                                        Il,
                                        Isum)
    
    # Show calculated intensity difference vs displacement and get linear fit coefficients of the calibration:
    # without normalization
    plot_shift_curves(k_px_um=k_px_um,
                      Il=Il,
                      Iz=Iz,
                      Isum=Isum,
                      x_shift=x_shift,
                      normalization=False,
                      shift_vs_sig=shift_vs_sig)
    k_x, b_x, k_y, b_y = calc_calib_line(x_shift=x_shift,
                                         y_shift=x_shift,
                                         k_px_um=k_px_um,
                                         Il=Il,
                                         Iz=Iz,
                                         normalization=False,
                                         shift_vs_sig=shift_vs_sig)
    # with normalization
    plot_shift_curves(k_px_um=k_px_um,
                      Il=Il,
                      Iz=Iz,
                      Isum=Isum,
                      x_shift=x_shift,
                      normalization=True,
                      shift_vs_sig=shift_vs_sig)
    k_x_norm, b_x_norm, k_y_norm, b_y_norm = calc_calib_line(x_shift=x_shift,
                                         y_shift=x_shift,
                                         k_px_um=k_px_um,
                                         Il=Il,
                                         Iz=Iz,
                                         Isum=Isum,
                                         normalization=True,
                                         shift_vs_sig=shift_vs_sig)
    return k_x, b_x, k_y, b_y, k_x_norm, b_x_norm, k_y_norm, b_y_norm



