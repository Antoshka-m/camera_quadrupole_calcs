from scipy.stats import linregress

def calc_calib_line(x_shift, y_shift, k_px_um, Il, Iz, Isum=None, normalization=False, shift_vs_sig=True):
    """
    Calculate coefficient of calibration line shift to signal of signal to shift.
    Calculations done only on lateral axis (LR signal)
    Shifts taken in nm units.
    
    Parameters
    -------
    x_shift: np-array
        dx shifts of image
    y_shift: np.array
        dy shifts of image
    k_px_um: float
        px to um scale (from px size)
    Il: np.array
        LR (lateral) intensity difference signal
    Iz: np.array
        TB (vertical) intensity difference signal
    Isum: np.array
        sum intensity signal 
    normalization: boolean
        if True, doing calculation for signal normalized over Isum
        
    Returns
    -------
    k: float
        slope of linear fit
    b: float
        intercept of linear fit
    """
    x_shift_nm = (x_shift/k_px_um)*1E3
    y_shift_nm = (y_shift/k_px_um)*1E3
    if normalization:
        if shift_vs_sig:
            #calculate k, b for shift vs intensity, normalized over Isum for x
            k_x, b_x, _, _, _ = linregress(Il/Isum, x_shift_nm) 
            k_y, b_y, _, _, _ = linregress(Iz/Isum, y_shift_nm) 
        else:
            #k, b for intensity vs shift, normalized over Isum
            k_x, b_x, _, _, _ = linregress(x_shift_nm, Il/Isum) 
            k_y, b_y, _, _, _ = linregress(y_shift_nm, Iz/Isum) 
    else: # k, b for non-normalized intensity signal
        if shift_vs_sig:
            k_x, b_x, _, _, _ = linregress(Il, x_shift_nm)
            k_y, b_y, _, _, _ = linregress(Iz, y_shift_nm)
        else:
            k_x, b_x, _, _, _ = linregress(x_shift_nm, Il)
            k_y, b_y, _, _, _ = linregress(y_shift_nm, Iz)
    if b_x>=0:
        print('The calibration line for x axis (a.u. to nm) has formula {}*x+{}\n'.format(round(k_x, 3), round(b_x, 3)))
    else:
        print('The calibration line for x axis (a.u. to nm) has formula {}*x{}\n'.format(round(k_x, 3), round(b_x, 3)))
    if b_y>=0:
        print('The calibration line for y axis (a.u. to nm) has formula {}*x+{}\n'.format(round(k_y, 3), round(b_y, 3)))
    else:
        print('The calibration line for y axis (a.u. to nm) has formula {}*x{}\n'.format(round(k_y, 3), round(b_y, 3)))
    return k_x, b_x, k_y, b_y
