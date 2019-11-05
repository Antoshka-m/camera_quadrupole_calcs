import matplotlib.pyplot as plt
"""
    Showing frame of the video and calibration lines for the 
    intensity \ shift calibration curve script
"""

def plot_im_w_quadrants(frame, centroid, fig_title):
    """
    Plot single image and showing the quadrants
    
    Parameters
    -------
    image: np-array
        grayscale image
    centroid: tuple
        coordinates of the center of quadrants
    """

    plt.figure()
    plt.title(fig_title)
    plt.imshow(frame, cmap=plt.cm.gray)
    plt.plot(centroid[1], centroid[0], "x", c="red")
    plt.axvline(x=centroid[1], c="red", linewidth=1)
    plt.axhline(y=centroid[0], c="red", linewidth=1)
    plt.show()


def plot_shift_curves(k_px_um, Il, Iz, Isum, x_shift, normalization=False, shift_vs_sig=True):
    """
    Plot Intensity difference (a.u.) vs shift (in px and nm)
    
    Parameters
    -------
    k_px_um: float
        px to um scale (from px size)
    Il: np.array
        LR (lateral) intensity difference signal
    Iz: np.array
        TB (vertical) intensity difference signal 
    Isum: np.array
        sum intensity signal 
    normalization: boolean
        if True, plots signal normalized over Isum
    """
    # x_shift = np.array(x_shift)
    # Il=np.array(Il)
    # Iz=np.array(Iz)
    # Isum=np.array(Isum)
    plt.figure()
    plt.title('shift in px, calibration line')
    plt.grid()
    if normalization:  # plots with intensity difference normalized over Isum
        if shift_vs_sig: # plot shift vs intensity
            plt.plot(Il/Isum, x_shift, label='lateral', linestyle='-', marker='o')
            plt.plot(Iz/Isum, x_shift, label='vertical', linestyle='-', marker='o')
            plt.xlabel('Intensity difference normalized, a.u.')
            plt.ylabel('Shift, px')
        else: # or plot intensity vs shift
            plt.plot(x_shift, Il/Isum, label='lateral', linestyle='-', marker='o')
            plt.plot(x_shift, Iz/Isum, label='vertical', linestyle='-', marker='o')
            plt.ylabel('Intensity difference normalized, a.u.')
            plt.xlabel('Shift, px')
    else: # plots with intensity difference without normalization
        if shift_vs_sig: # plot shift vs intensity
            plt.plot(Il, x_shift, label='lateral', linestyle='-', marker='o')
            plt.plot(Iz, x_shift, label='vertical', linestyle='-', marker='o')
            plt.xlabel('Intensity difference, a.u.')
            plt.ylabel('Shift, px')
        else: # plot intensity vs shift
            plt.plot(x_shift, Il, label='lateral', linestyle='-', marker='o')
            plt.plot(x_shift, Iz, label='vertical', linestyle='-', marker='o')
            plt.ylabel('Intensity difference, a.u.')
            plt.xlabel('Shift, px')
    plt.legend()
    plt.figure() # second figure has the same plots, but x_shift scaled to nm
    plt.title('shift in nm, calibration line')
    plt.grid()
    x_shift_nm = (x_shift/k_px_um)*1E3
    if normalization:  # plots with intensity difference normalized over Isum
        if shift_vs_sig: # plot shift vs intensity
            plt.plot(Il/Isum, x_shift_nm, label='lateral', linestyle='-', marker='o')
            plt.plot(Iz/Isum, x_shift_nm, label='vertical', linestyle='-', marker='o')
            plt.xlabel('Intensity difference normalized, a.u.')
            plt.ylabel('Shift, nm')
        else: # or plot intensity vs shift
            plt.plot(x_shift_nm, Il/Isum, label='lateral', linestyle='-', marker='o')
            plt.plot(x_shift_nm, Iz/Isum, label='vertical', linestyle='-', marker='o')
            plt.ylabel('Intensity difference normalized, a.u.')
            plt.xlabel('Shift, nm')
    else: # plots with intensity difference without normalization
        if shift_vs_sig: # plot shift vs intensity
            plt.plot(Il, x_shift_nm, label='lateral', linestyle='-', marker='o')
            plt.plot(Iz, x_shift_nm, label='vertical', linestyle='-', marker='o')
            plt.xlabel('Intensity difference, a.u.')
            plt.ylabel('Shift, nm')
        else: # plot intensity vs shift
            plt.plot(x_shift_nm, Il, label='lateral', linestyle='-', marker='o')
            plt.plot(x_shift_nm, Iz, label='vertical', linestyle='-', marker='o')
            plt.ylabel('Intensity difference, a.u.')
            plt.xlabel('Shift, nm')
    plt.legend()
    plt.show()
