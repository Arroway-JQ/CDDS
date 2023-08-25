import numpy as np
import scipy.io as sio
import math
import itertools
import matplotlib.pyplot as plt
"""
def D_boundary(SNR=100, b_3=20/1000, b_last=5000/1000):
    
    Generate T2 boundaries for SAME-ECOS analysis.

    Args:
        SNR (float, optional): signal to noise ratio. Defaults to 100.
        b_3 (float, optional): 3rd echo time (ms). Defaults to 30.
        b_last (float, optional): last echo time (ms). Defaults to 320.

    Returns:
        T2_min, T2_max: T2 lower and upper boundaries (ms)
    

    D_min = -b_3 / np.log(1 / SNR)
    D_max = -b_last / np.log(1 / SNR)

    return D_min, D_max
"""


"""

def T2_lower_boundary(SNR=100, num_T2=1, echo_1=10):
    
    Determine T2_min if there are multiple T2 component.

    Args:
        SNR (float, optional): signal to noise. Defaults to 100.
        num_T2 (int, optional): number of T2 components. Defaults to 1.
        echo_1 (float, optional): first echo time (ms). Defaults to 10.

    Returns:
        T2_min: T2 lower boundary (ms)
    

    T2_min = np.rint(-echo_1 * (2 * num_T2 + 1) / np.log(1 / SNR))
    return T2_min
"""

def D_components_resolution_finite_domain(SNR=100, D_min=7, D_max=1000):
    """
    Numerically determine the maximum number of T2 components that can be resolved for a given SNR at a certain T2 range.

    Args:
        SNR (float, optional): signal to noise ratio. Defaults to 100.
        D_min (float, optional): D lower boundary (mum2/ms). Defaults to 0.0074.
        D_max (float, optional): D upper boundary (mum2/ms). Defaults to 5.11.

    Returns:
        M: the maximum number of D components that can be resolved.
    """

    M = 1
    f = -1
    while f < 0:
        M = M + 0.01
        f = M / np.log(D_max / D_min) * np.sinh(np.pi ** 2 * M / np.log(D_max / D_min)) - (SNR / M) ** 2
    return M


def resloution_limit(D_min=3, D_max=3000 , M=4):
    """
    Finite domain T2 resolution calculation.

    Args:
        D_min (float, optional): D lower boundary (ms). Defaults to 0.0032.
        D_max (float, optional): D upper boundary (ms). Defaults to 10**0.48.
        M (float, optional): maximum number of resolvable D components. Defaults to 4.

    Returns:
        resolution: the D resolution.
    """

    resolution = (D_max / D_min) ** (1 / M)
    return resolution


def d_basis_generator(D_min=3, D_max=3000, num_basis_D=40):
    """
    Generate T2 basis set.

    Args:
        D_min (float, optional): T2 lower boundary (ms). Defaults to 7.
        D_max (float, optional): T2 upper boundary (ms). Defaults to 1000.
        num_basis_D (int, optional): number of basis ds. Defaults to 40.

    Returns:
        d_basis: generated basis ds (ms).
    """

    d_basis = np.geomspace(D_min, D_max, num_basis_D)
    return d_basis

"""
def analysis_boundary_condition(SNR=100, echo_1=10, T2_max=2000):
    
    Analytically determines the boundary condition for the analysis according to experimental conditions.
    For n T2 components, the residual signal of the shortest T2 component on the (2n+1)th echo has to be larger than the noise.
    The boundary conditions yield to the resolution limit formula.
    Returns the maximum number of T2 components allowed and a list for T2_min. Use with Caution!

    Args:
        SNR (float, optional): signal to noise ratio. Defaults to 100.
        echo_1 (float, optional): 1st echo time (ms). Defaults to 10.
        T2_max (float, optional): T2 upper limit (ms). Defaults to 2000.

    Returns:
        num_T2-1, T2_min_list: maximum resolvable T2 components, a list of T2 lower boundaries
    

    T2_min_list = []
    num_T2 = 1
    while True:
        T2_min = -echo_1 * (2 * num_T2 + 1) / np.log(1 / SNR)
        M = D_components_resolution_finite_domain(SNR, T2_min, T2_max)
        if M <= num_T2:
            break
        else:
            num_T2 = num_T2 + 1
            T2_min_list.append(T2_min)
    return num_T2 - 1, np.ceil(T2_min_list)
"""

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    value = array[idx]
    return idx,array[idx]

def D_location_generator_v3(Dlist,D_min=3, D_max=3000, num_D=3, M_max=5, scale='log'):
    """
    Randomly generate T2 locations.
    Generated T2 peaks should have uniform distribution (scale = linear or log) at the range of [T2_min, T2_max]

    Args:
        Dlist(float,must): accessed from the DWI_decay file.
        D_min (float, optional): D lower boundary (ms). Defaults to 0.074.
        D_max (float, optional): D upper boundary (ms). Defaults to 10**0.48.
        num_T2 (int, optional): number of T2 component. Defaults to 3.
        M_max (int, optional): global maximal number of resolvable D components. Defaults to 5.
        scale (str, optional): logarithmic or linear scale over the D range. Defaults to 'log'.

    Returns:
        D_locations: An array of D locations (ms)
    """

    D_location = np.zeros(num_D)
    D_locationID = np.zeros(num_D)
    resolution = resloution_limit(D_min, D_max, M_max)
    D_low = D_min
    for i in range(num_D):
        ### uniform distribution on linear scale
        if scale == 'linear':
            D_location[i] = np.random.randint(D_low, D_max / (resolution ** (num_D - i - 1)))
            ### uniform distribution on logarithmic scale (https://stackoverflow.com/questions/43977717/how-do-i-generate-log-uniform-distribution-in-python/43977980)
        if scale == 'log':
            idx, D_lownear = find_nearest(Dlist,D_low)
            D_low = D_lownear
            D_locationID[i] = np.random.randint(idx,2000)
            D_locationID = D_locationID.astype(int)
            D_location[i] = Dlist[D_locationID[i]]
            #D_location[i] = np.exp(np.random.uniform(np.log(D_low), np.log(D_max / (resolution ** (num_D - i - 1)))))
        D_low = D_location[i] * resolution  # set the new lower boundary
    #T2_location = T2_location.astype(int)
    return D_location,D_locationID


def D_resolution_examiner(D_location, resolution):
    """
    Examine whether two T2 peak locations obey the resolution limit.

    Args:
        T2_location (array): an array of T2 locations (ms)
        resolution (float): the resolution limit

    Returns:
        TF: True or False
    """

    TF = True
    for x, y in itertools.combinations(D_location, 2):
        if x / y > 1 / resolution and x / y < resolution:
            TF = False
            break
    return TF


def required_frequency(D_location):
    """
    Calculate the required frequency w to construct the generated T2 peaks in the T2 domain.

    Args:
        D_location (array): an array of T2 locations (ms)

    Returns:
        ratio_min, frequency: the minimal ratio between adjacent components, required frequency.
    """

    D_location_sort = np.sort(D_location)
    ratio_min = (D_location_sort[1:] / D_location_sort[:-1]).min()
    frequency = np.pi / np.log(ratio_min)
    return ratio_min, frequency


def minimum_amplitude_calculator(SNR, frequency):
    """
    Calculate the minimum allowable amplitude for all T2 components at a given SNR.

    Args:
        SNR (float): signal to noise ratio
        frequency (float): required frequency of the T2 components

    Returns:
        minimum_amp_T2_peak: the minimal amplitude of T2 components
    """

    amp_noise = 1 / SNR
    minimum_amp_D_peak = amp_noise * np.sqrt(frequency / np.pi * np.sinh(np.pi * frequency))
    return minimum_amp_D_peak


def D_amplitude_generator_v3(num_D, minimum_amplitude):
    """
    Randomly generate normalized T2 peak amplitude.
    Generated amplitude should have uniform distribution at range [minimum_amplitude, 1-(num_T2-1)*minimum_amplitude]

    Args:
        num_D (int): number of D components
        minimum_amplitude (float): the minimal amplitude

    Returns:
        D_amplitude: an array of D amplitudes
    """

    D_amplitude = np.zeros(num_D)
    remainder = 1
    for i in range(num_D - 1):
        D_amplitude[i] = np.random.uniform(minimum_amplitude, remainder - (num_D - i - 1) * minimum_amplitude, 1)
        remainder = 1 - D_amplitude.sum()
    D_amplitude[-1] = remainder
    D_amplitude = D_amplitude / D_amplitude.sum()  # normalization
    D_amplitude = np.squeeze(D_amplitude)
    if D_amplitude.shape[0] == 1:
        D_amplitude = D_amplitude.reshape(1, )
        ### amplitude array has a descending trend so it needs to be shuffled
    np.random.shuffle(D_amplitude)
    return D_amplitude


def metrics_extraction_v3(T2_location, T2_amplitude, MW_low_cutoff=0, MW_high_cutoff=40, IEW_low_cutoff=40,
                          IEW_high_cutoff=200):
    """
    This function extracts five metrics: myelin water fraction (MWF), MW geometric mean T2 (MWGMT2), IEWF, IEWGMT2, GMT2.

    Args:
        T2_location (array): an array of T2 locations (ms)
        T2_amplitude (array): an array of T2 amplitudes
        MW_low_cutoff (float, optional): myelin water lower boundary (ms). Defaults to 0.
        MW_high_cutoff (float, optional): myelin water upper boundary (ms). Defaults to 40.
        IEW_low_cutoff (float, optional): IE water lower boundary (ms). Defaults to 40.
        IEW_high_cutoff (float, optional): IE water upper boundary (ms). Defaults to 200.

    Returns:
        MWF, MWGMT2, IEWF, IEWGMT2, GMT2: myelin water fraction (MWF), MW geometric mean T2 (MWGMT2), IEWF, IEWGMT2, GMT2
    """

    ### get the location index of MW and IEW
    MW_loc = (T2_location >= MW_low_cutoff) & (T2_location <= MW_high_cutoff)
    IEW_loc = (T2_location >= IEW_low_cutoff) & (T2_location <= IEW_high_cutoff)
    ### calculate MWF and IEWF
    MWF = T2_amplitude[MW_loc].sum() / T2_amplitude.sum()
    IEWF = T2_amplitude[IEW_loc].sum() / T2_amplitude.sum()
    ### calculate weighted geometric mean using equation in https://en.wikipedia.org/wiki/Weighted_geometric_mean
    if MWF == 0:
        MWGMT2 = np.nan
    else:
        MWGMT2 = np.exp(np.dot(T2_amplitude[MW_loc], np.log(T2_location[MW_loc])) / T2_amplitude[MW_loc].sum())
    if IEWF == 0:
        IEWGMT2 = np.nan
    else:
        IEWGMT2 = np.exp(np.dot(T2_amplitude[IEW_loc], np.log(T2_location[IEW_loc])) / T2_amplitude[IEW_loc].sum())
    ### calculate the overall weighted geometric mean
    GMT2 = np.exp(np.dot(T2_amplitude, np.log(T2_location)) / T2_amplitude.sum())
    return MWF, MWGMT2, IEWF, IEWGMT2, GMT2


def load_decay_lib(file_path):
    """
    Load the calculated decay library from .mat file.

    Args:
        file_path (str): the file path to the .mat file

    Returns:
        decay_lib: the loaded decay library
    """

    decay_lib = sio.loadmat(file_path)
    decay_lib = decay_lib['Ddecay']
    return decay_lib


def produce_decay_from_lib(decay_lib, D_locationID, D_amplitude):
    """
    Generate the decay curve generation from the decay library: weighted sum of the selected T2 components.

    Args:
        decay_lib (array): the loaded decay library
        D_locationID (int): an array of the id of the D_locations
        D_amplitude (int): an array of D amplitudes


    Returns:
        decay_curve: the produced decay curve
    """

    decay_curve = np.sum(decay_lib[D_locationID, :] * D_amplitude.reshape(D_amplitude.shape[0], 1), axis=0)
    return decay_curve


def signal_with_noise_generation_phase_rotation(signal, SNR):
    """
    1. Project pure signal to real and imaginary axis according to a randomly generated phase factor.
    2. Generate noise (normal distribution on real and imaginary axis)
    3. Noise variance is determined by SNR (Rayleigh noise floor).

    Args:
        signal (array): the decay signal
        SNR (float): signal to noise ratio

    Returns:
        signal_with_noise: the signal with added noise on both real and imaginary axis
    """

    phase = math.pi / 2 * np.random.rand()
    signal_real = signal * math.cos(phase)
    signal_imaginary = signal * math.sin(phase)
    Rayleigh_noise_variance = 1 / (SNR * math.sqrt(math.pi / 2))
    noise_real = np.random.normal(0, Rayleigh_noise_variance, signal.shape[0])
    noise_imaginary = np.random.normal(0, Rayleigh_noise_variance, signal.shape[0])
    signal_with_noise = ((signal_real + noise_real) ** 2 + (signal_imaginary + noise_imaginary) ** 2) ** 0.5
    return signal_with_noise


def train_label_generator(D_location, D_amplitude, d_basis):
    """
    This function takes randomly generated d peak locations and amplitudes as inputs, and uses basis ds to represent these D peaks.
    Each peak is embedded by two nearest basis Ds.

    Args:
        D_location (array): an array of D locations (ms)
        D_amplitude (array): an array of D amplitudes
        d_basis (array): the basis ds (mum^2/ms)

    Returns:
        train_label: the D spectrum depicted by the basis ds
    """

    ### create multi-dimensional placeholder (each dimension for each peak)
    train_label = np.zeros([D_location.shape[0], d_basis.shape[0]])
    ### iterate through each peak and find the nearest couple of d basis and assign weighting factors
    for i in range(D_location.shape[0]):
        for j in range(d_basis.shape[0]):
            if abs(d_basis[j] - D_location[i]) < 0.000000001:
                train_label[i, j] = D_amplitude[i]
            elif d_basis[j - 1] < D_location[i] and d_basis[j] > D_location[i]:
                train_label[i, j - 1] = (d_basis[j] - D_location[i]) / (d_basis[j] - d_basis[j - 1]) * D_amplitude[i]
                train_label[i, j] = (D_location[i] - d_basis[j - 1]) / (d_basis[j] - d_basis[j - 1]) * D_amplitude[i]
                ### return one dimensional train label
    return train_label.sum(axis=0)


def train_label_generator_gaussian_embedding(D_location, D_amplitude, D_min, D_max, d_basis, sigma=1):
    """
    This function takes randomly generated d peak locations and amplitudes as inputs, and uses basis ds to represent these D peaks.
    Each peak generates a gaussian function centered at its peak location (in log space), and then embedded by all basis Ds

    Args:
        D_location (array): an array of D locations (ms)
        D_amplitude (array): an array of D amplitude
        D_min (float): the D lower boundary (ms)
        D_max (float): the D upper boundary (ms)
        d_basis (array): the basis ds (ms)
        sigma (float, optional): variance of the Gaussian peaks. Defaults to 1.

    Returns:
        train_label: the D spectrum depicted by basis ds
    """

    ### create multi-dimensional placeholder (each dimension for each peak)
    train_label = np.zeros([D_location.shape[0], d_basis.shape[0]])
    ### iterate through each peak and assign weighting factors to d_basis according to normal distribution
    for i in range(D_location.shape[0]):
        train_label[i, :] = gaussian_embedding_log_scale(D_location[i], D_min=D_min, D_max=D_max,d_basis=d_basis, sigma=sigma) * D_amplitude[i]
        ### return one dimensional train label
    return train_label.sum(axis=0)


def gaussian_embedding_log_scale(peak, D_min, D_max, d_basis, sigma):
    """
    This function takes one d delta peak as inputs, and returns a normalized gaussian weighted d_basis labels on log scale.

    Args:
        peak (float): one D location
        D_min (float): D lower boundary (ms)
        D_max (float): D upper boundary (ms)
        d_basis (array): basis ds (ms)
        sigma (float): variance of the Gaussian peak.

    Returns:
        d_basis_weights_scaled: the normalized Gaussian peaks
    """

    base = (D_max / D_min) ** (1 / d_basis.shape[0])
    d_basis_index = np.arange(d_basis.shape[0])
    peak_index = np.log(peak / D_min) / np.log(base)
    d_basis_weights = 1 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-(d_basis_index - peak_index) ** 2 / (2 * sigma ** 2))
    d_basis_weights[d_basis_weights < 1e-7] = 0
    d_basis_weights_scaled = d_basis_weights / d_basis_weights.sum()
    return d_basis_weights_scaled


def produce_training_data(decay_lib,
                          Dlist,
                          realizations=10000,
                          SNR_boundary_low=25,
                          SNR_boundary_high=500,
                          b_3 = 20/1000,
                          b_last = 5000/1000,
                          b_train_num = 22,
                          num_D_basis=50,
                          peak_width=1,
                          D_min_universal=None,
                          D_max_universal=None,
                          exclude_M_max=True):
    """
    Produce training data via SAME-ECOS simulation pipeline (use a single cpu core).

    Args:
        decay_lib (array): the decay library
        Dlist (array): the D libarary of the decay library
        realizations (int, optional): the number of simulation realizations. Defaults to 10000.
        SNR_boundary_low (float, optional): lower boundary of SNR. Defaults to 50.
        SNR_boundary_high (float, optional): upper boundary of SNR. Defaults to 800.
        num_D_basis (int, optional): the number of basis ds. Defaults to 40.
        peak_width (float, optional): the variance of the gaussian peak. Defaults to 1.
        D_min_universal (float, optional): the overall minimal T2 (ms) of the analysis. Defaults to calculate on the fly if None is given.
        D_max_universal (float, optional): the overall maximal T2 (ms) of the analysis. Defaults to to 2000ms if None is given.
        exclude_M_max (bool, optional): exclude the M_max if True. Defaults to True.

    Returns:
        data: dictionary collection of the produced training data
    """

    ### Define T2 range, maximum number (M_max) of T2 peaks at the highest SNR, allowable number (N) of T2 peaks for simulation
    if D_min_universal == None:
        D_min_universal = 0.003 #D_boundary(SNR_boundary_high, b_3,b_last)  ## Lower boundary is determined by the highest SNR
    if D_max_universal == None:
        D_max_universal = 3  ## empirically determined
    d_basis = d_basis_generator(D_min_universal, D_max_universal, num_D_basis)
    M_max = int(np.floor(D_components_resolution_finite_domain(SNR_boundary_high, D_min_universal,D_max_universal)))  ## M at highest SNR
    # resolution_max = resloution_limit(D_min_universal, D_max_universal, M_max) ## resolution at highest SNR
    if exclude_M_max == True:
        N = M_max - 1  ## for simulation M_max is excluded
    else:
        N = M_max
    ### Create placeholders for memory efficiency
    D_location_all = np.zeros([realizations, N])
    D_locationID_all = np.zeros([realizations, N])
    D_amplitude_all = np.zeros([realizations, N])
    decay_curve_all = np.zeros([realizations, b_train_num])
    decay_curve_with_noise_all = np.zeros([realizations, b_train_num])
    train_label_all = np.zeros([realizations, num_D_basis])
    train_label_gaussian_all = np.zeros([realizations, num_D_basis])
    num_D_SNR_all = np.zeros([realizations, 2])
    ### For each realization
    for i in range(realizations):
        ### Randomly determine the SNR, the minimum D, the number of Ds (must < M), and the flip angle FA.
        # SNR = 100 ## for fixed SNR
        SNR = np.random.randint(SNR_boundary_low, SNR_boundary_high)  # only one number
        D_min = D_min_universal#D_boundary(SNR, b_3, b_last)
        D_max = D_max_universal
        M = np.floor(D_components_resolution_finite_domain(SNR, D_min, D_max))
        if exclude_M_max == True:
            N_choice = np.arange(1, M-1)  # a list from 1 to M-1
        else:
            N_choice = np.arange(1, M + 1)
        weight = N_choice ** 0.2  ## weighting factor for each choice
        num_D = int(np.random.choice(N_choice, p=weight / weight.sum()))  # randomly choose a number from list 'N_choice'
        ### Calculate the resolution limit
        resolution = resloution_limit(D_min, D_max, M)
        ### Randomly generate T2 peak location with respect to resolution limit.
        # T2_location = T2_location_generator_v3(T2_min, T2_max, num_T2, num_d_basis, resolution, log_cutoff=10, smooth=False)
        D_location,D_locationID = D_location_generator_v3(Dlist,D_min, D_max, num_D, M_max, scale='log')
        while D_resolution_examiner(D_location,resolution) == False:  # examine whether T2 location obeys the resolution
            D_location, D_locationID = D_location_generator_v3(Dlist, D_min, D_max, num_D, M_max, scale='log')
            ### Randomly generate D peak amplitude. When two or more peaks, minimal detectable amplitude is calculated
        if num_D == 1:
            D_amplitude = np.array([1.0])
        else:
            _, frequency = required_frequency(D_location)
            minimum_amplitude = minimum_amplitude_calculator(SNR, frequency)  # equation(8)
            D_amplitude = D_amplitude_generator_v3(num_D, minimum_amplitude)
        ### ！！！！！！Decay curve generation (weighted sum of each D component)！！！！！！！
        decay_curve = produce_decay_from_lib(decay_lib, D_locationID, D_amplitude)
        ### Add noise to decay curve
        decay_curve_with_noise = signal_with_noise_generation_phase_rotation(signal=decay_curve, SNR=SNR)
        ### D basis set embedding (nearest d_basis neighbors)
        train_label = train_label_generator(D_location, D_amplitude, d_basis)
        ### D basis set embedding (gaussian peaks)
        train_label_gaussian = train_label_generator_gaussian_embedding(D_location, D_amplitude, D_min_universal,D_max_universal, d_basis, peak_width)
        ### Extract metrics (use d_basis and train label instead of D_location and D_amplitude to prevent basis set embedding error)
        # MWF, MWGMD, IEWF, IEWGMD, GMD = metrics_extraction_v3(d_basis, train_label, MW_low_cutoff, MW_high_cutoff, IEW_low_cutoff, IEW_high_cutoff)
        ### Ground truth metrics (use tD_location and D_amplitude)
        # MWF_GT, MWGMD_GT, IEWF_GT, IEWGMD_GT, GMD_GT = metrics_extraction_v3(D_location, D_amplitude, MW_low_cutoff, MW_high_cutoff, IEW_low_cutoff, IEW_high_cutoff)
        ### Pad D_location and D_amplitude to have uniform size
        D_location = np.pad(D_location, [(0, N - int(num_D))], mode='constant', constant_values=0)  #填充
        D_amplitude = np.pad(D_amplitude, [(0, N - int(num_D))], mode='constant', constant_values=0)
        ### Store generated parameters in placeholders
        D_location_all[i, :] = D_location
        #D_locationID_all[i,:] = D_locationID
        D_amplitude_all[i, :] = D_amplitude
        decay_curve_all[i, :] = decay_curve
        decay_curve_with_noise_all[i, :] = decay_curve_with_noise
        train_label_all[i, :] = train_label
        train_label_gaussian_all[i, :] = train_label_gaussian
        num_D_SNR_all[i, :] = num_D, SNR
    ### return a data dict
    data = {'D_location': D_location_all,
            'D_amplitude': D_amplitude_all,
            'decay_curve': decay_curve_all,
            'decay_curve_with_noise': decay_curve_with_noise_all,
            'train_label': train_label_all,
            'train_label_gaussian': train_label_gaussian_all,
            'num_D_SNR': num_D_SNR_all,
            }
    return data


def mp_yield_training_data(func_produce_training_data,
                           decay_lib,
                           Dlist,
                           realizations,
                           ncores,
                           SNR_boundary_low=25,
                           SNR_boundary_high=500,
                           b_3=20/1000,
                           b_last=5000/1000,
                           b_train_num=22,
                           num_d_basis=50,
                           peak_width=1,
                           D_min_universal=None,
                           D_max_universal=None,
                           exclude_M_max=True):
    """
    Use multiple cpu cores to accelerate training data production using multiprocessing package.

    Args:
        func_produce_training_data (function): the function to produce training data using a single cpu core.
        decay_lib (array): the decay library.
        realizations (int): the number of simulation realizations.
        ncores (int): number of cpu cores to use.
        SNR_boundary_low (float, optional): lower boundary of SNR. Defaults to 50.
        SNR_boundary_high (float, optional): upper boundary of SNR. Defaults to 800.
        echo_3 (float, optional): the 3rd echo time (ms). Defaults to 30.
        echo_last (float, optional): the last echo time (ms). Defaults to 320.
        echo_train_num (int, optional): the number of echoes in the echo train. Defaults to 32.
        num_d_basis (int, optional): the number of basis ds. Defaults to 40.
        FA_min (float, optional): the minimal refocusing flip angle for simulation. Defaults to 50.
        peak_width (float, optional): the variance of the gaussian peak. Defaults to 1.
        D_min_universal (float, optional): the overall minimal D (ms) of the analysis. Defaults to calculate on the fly if None is given.
        D_max_universal (float, optional): the overall maximal D (ms) of the analysis. Defaults to to 2000ms if None is given.
        exclude_M_max (bool, optional): exclude the M_max if True. Defaults to True.

    Returns:
        data_all: a data dictionary concatenated from all cpu cores
    """

    import multiprocessing as mp
    pool = mp.Pool(processes=ncores)
    ### distribute job to each cpu core
    realizations_pool_list = [realizations // ncores] * ncores
    if realizations % ncores != 0:
        realizations_pool_list.append(realizations % ncores)
    data = pool.starmap(func_produce_training_data, [(decay_lib, Dlist, realizations,
                                                      SNR_boundary_low, SNR_boundary_high,
                                                      b_3, b_last, b_train_num,
                                                      num_d_basis, peak_width,
                                                      D_min_universal, D_max_universal,
                                                      exclude_M_max)
                                                     for realizations in realizations_pool_list])
    pool.close()
    pool.join()
    ### concatenate data calculated from each cpu core
    keys = data[0].keys()
    data_all = {key: None for key in keys}
    for key in keys:
        data_all[key] = np.concatenate([data[x][key] for x in range(len(data))])
    return data_all


def plot_all_echoes(img, slice_num, rows, columns, fig_size=None, tight=True):
    """Plot all echoes (axis=-1) of the 4D image data at a given slice (axis=-2).

    Args:
        img (4D array): the 4D image data.
        slice_num (int): the slice number.
        rows (int): the number of subfigures to plot in each row.
        columns (int): the number of subfigures to plot in each column.
        fig_size (tuple, optional): the figure size. Defaults to None.
        tight (bool, optional): tight layout when plot. Defaults to True.
    """

    if fig_size != None:
        fig = plt.figure(figsize=fig_size)
    else:
        fig = plt.figure()
    for i in range(img.shape[-1]):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img[:, :, slice_num, i])
        plt.title('echo {}'.format(i + 1))
        plt.axis('off')
    if tight:
        plt.tight_layout()
    plt.show()


def NN_predict_4D_decay(decay_data, NN_model):
    """
    Use trained neural network to make predictions on 4D image data.

    Args:
        decay_data (4D array): 4D image data with the last dimension of the echo train.
        NN_model (model): the pre-trained neural network model.

    Returns:
        4D array: the predicted spectrum of each voxel
    """
    ### flat voxels, and normalize to the first echo
    decay_flat = decay_data.reshape(
        decay_data.shape[0] * decay_data.shape[1] * decay_data.shape[2],
        decay_data.shape[3])
    decay_flat_norm = decay_flat / (decay_flat[:, 0].reshape(
        decay_data.shape[0] * decay_data.shape[1] * decay_data.shape[2], 1))
    ### use trained model to predict the spectrum
    NN_predict_spectrum_flat = NN_model.predict(decay_flat_norm)
    ### reshape the flat spectrum back to 4D array
    NN_predict_spectrum = NN_predict_spectrum_flat.reshape(
        decay_data.shape[0], decay_data.shape[1], decay_data.shape[2],
        NN_predict_spectrum_flat.shape[1])

    return NN_predict_spectrum


def quantitative_map_production(d_basis,
                                spectrum,
                                MW_low_cutoff=0,
                                MW_high_cutoff=40,
                                IEW_low_cutoff=40,
                                IEW_high_cutoff=200):
    """
    This function produce 5 quantitative maps from predicted spectra: MWF, MWGMD, IEWF, IEWGMD, GMD.
    Spectrum in a data shape such as (240, 240, 40, 40) with the last dimension indicating the number of basis ds.
    This function is calling another function 'metric_extraction_v3'.

    Args:
        d_basis (array): basis ds (ms).
        spectrum (4D array): the predicted spectrum of each image voxel.
        MW_low_cutoff (float, optional): the lower boundary of myelin water (ms). Defaults to 0.
        MW_high_cutoff (float, optional): the upper boundary of myelin water (ms). Defaults to 40.
        IEW_low_cutoff (float, optional): the lower boundary of IE water (ms). Defaults to 40.
        IEW_high_cutoff (float, optional): the upper boundary of IE water (ms). Defaults to 200.

    Returns:
        4D array: the last dimension in order: MWF, MWGMD, IEWF, IEWGMD, GMD
    """

    spectrum_flat = spectrum.reshape(
        spectrum.shape[0] * spectrum.shape[1] * spectrum.shape[2],
        spectrum.shape[3])
    NN_predict_metrics_flat = np.zeros((spectrum_flat.shape[0], 5))
    for item in range(spectrum_flat.shape[0]):
        NN_predict_metrics_flat[item, :] = metrics_extraction_v3(
            d_basis, spectrum_flat[item, :], MW_low_cutoff, MW_high_cutoff,
            IEW_low_cutoff, IEW_high_cutoff)
    NN_predict_metrics = NN_predict_metrics_flat.reshape(
        spectrum.shape[0], spectrum.shape[1], spectrum.shape[2],
        NN_predict_metrics_flat.shape[1])
    return NN_predict_metrics


def plot_all_slice(maps, nrow, ncol, vmin=None, vmax=None, figsize=(30, 12), fontsize=20, cfontsize=20, cshrink=0.3,
                   cpad=0.05, clocation='bottom'):
    """
    Plot all slices of a 3D image.

    Args:
        maps (3D array): 3D image data.
        nrow (int): number of rows
        ncol (int): number of columns
        vmin (float, optional): minimal intensity. Defaults to None.
        vmax (float, optional): maximal intensity. Defaults to None.
        figsize (tuple, optional): figure size. Defaults to (30, 12).
        fontsize (float, optional): the font size. Defaults to 20.
        cfontsize (float, optional): colorbar font size. Defaults to 20.
        cshrink (float, optional): colorbar shrink factor. Defaults to 0.3.
        cpad (float, optional): padding between figure and colorbar. Defaults to 0.05.
        clocation (str, optional): colorbar location. Defaults to 'bottom'.
    """

    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    for i, ax in enumerate(axs.flat):
        if (vmin and vmax) is not None:
            img = ax.imshow(maps[:, :, i], vmin=vmin, vmax=vmax)
        else:
            img = ax.imshow(maps[:, :, i])
        ax.set_title('slice {}'.format(i + 1), fontsize=fontsize)
        ax.axis('off')
    plt.tight_layout()
    cbar = fig.colorbar(img, ax=axs, shrink=cshrink,
                        location=clocation, pad=cpad)
    cbar.ax.tick_params(labelsize=cfontsize)
    plt.show()
