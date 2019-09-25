import numpy as np
import matplotlib.pyplot as plt
from skvideo.utils import rgb2gray
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


# Let's import first frame of videofile, and show it

file = get_filenames()
file = file[0] # get just single file instead of list
print('Importing file ', file)
frame = skvideo.io.vread(file, num_frames=1) # import just first frame
frame = rgb2gray(frame[0]) # get element instead of list, make grayscale
plt.figure()
plt.imshow(frame, cmap=plt.cm.gray)
plt.show()

# Compensate angle if its needed

finish=False
angle=0
while finish==False:
        angle, finish=rotate_image(frame, angle)
frame=rotate(frame, angle)

# Detect center of lightspot, show quadrants:

centroid=threshold_centroid(frame)
plot_im_w_quadrants(frame, centroid)

# Demonstrate how shifted image looks like

transform = AffineTransform(translation=(1, 0))
shifted = warp(frame, transform, mode='constant', preserve_range=True)
plot_im_w_quadrants(shifted, centroid)

# Shift images along x axis

shifted_im = []
x_shift = np.array([0.1*dx for dx in range(0, 11)])  # generate dx value for linear shift
for dx in x_shift:
    transform = AffineTransform(translation=(dx, 0))  # shift along lateral axis
    shifted_im.append(warp(frame, transform, mode='constant', preserve_range=True))

# Calculate the intensities

Il=np.array([])
Iz=np.array([])
Isum=np.array([])
for i in range(len(shifted_im)):
    Iz, Il, Isum = calc_intensities(shifted_im[i], centroid, Iz, Il, Isum)

# Show calculated intensity difference vs displacement and get linear fit coefficients of the calibration:

plot_shift_curves(k_px_um=1.36, Il=Il, Iz=Iz, Isum=Isum, x_shift=x_shift, normalization=False, shift_vs_sig=True)
k, b = calc_calib_line(x_shift=x_shift, k_px_um=1.36, Il=Il, normalization=False, shift_vs_sig=True)




