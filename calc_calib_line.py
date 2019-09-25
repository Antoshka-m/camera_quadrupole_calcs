from scipy.stats import linregress

def calc_calib_line(x_shift, k_px_um, Il, Isum=None, normalization=False, shift_vs_sig=True):
    """
    Calculate coefficient of calibration line shift to signal of signal to shift.
    Calculations done only on lateral axis (LR signal)
    Shifts taken in nm units.
    
    Parameters
    -------
    x_shift: np-array
        dx shifts of image
    k_px_um: float
        px to um scale (from px size)
    Il: np.array
        LR (lateral) intensity difference signal
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
    if normalization:
        if shift_vs_sig:
            k, b, _, _, _ = linregress(Il/Isum, x_shift_nm) #k, b for shift vs intensity, normalized over Isum
        else:
            k, b, _, _, _ = linregress(x_shift_nm, Il/Isum) #k, b for intensity vs shift, normalized over Isum
    else: # k, b for non-normalized intensity signal
        if shift_vs_sig:
            k, b, _, _, _ = linregress(Il, x_shift_nm)
        else:
            k, b, _, _, _ = linregress(x_shift_nm, Il)
    if b>0:
        print('The calibration line (a.u. to um) has formula {}*x+{}'.format(round(k, 3), round(b, 3)))
    else:
        print('The calibration line (a.u. to um) has formula {}*x{}'.format(round(k, 3), round(b, 3)))
    return k, b
