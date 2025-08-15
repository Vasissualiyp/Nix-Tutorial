import numpy as np
from time import sleep
import matplotlib
import matplotlib.pyplot as plt
from classy import Class
from scipy.optimize import curve_fit
import time
matplotlib.use('Agg') # Make plots interactive

class calculate_TF():
    """
    A class that incapsulates all variables and routines to calculate TF/PS.
    Defaults to output_type=4 (PeakPatch).
    """
    def __init__(self, output_type=4, log=False):
        """Initialize arrays of TF/PS, that will be derived"""
        self.allowed_output_types = [2, 4, 7, 13]
        self.reset_arrays()
        self.set_output_type(output_type)
        self.log = log
        self.epsilon = 1e-30

    # Initializations

    def reset_arrays(self):
        """Resets the arrays, containing TF/PS data"""
        self.kh = np.array([])
        self.delta_cdm = np.array([])
        self.delta_b = np.array([])
        self.delta_g = np.array([])
        self.delta_nu = np.array([])
        self.delta_num = np.array([])
        self.delta_tot = np.array([])
        self.delta_nonu = np.array([])
        self.delta_tot_nodm = np.array([])
        self.phi = np.array([])
        self.v_cdm = np.array([])
        self.v_b = np.array([])
        self.v_b_cdm = np.array([])
        self.norm = 0.0
        self.kk = np.array([])
        self.pk = np.array([])
        self.Trans = np.array([])
        self.pkchi = np.array([])
        self.dataset_names_music = ['delta_cdm', 'delta_b', 'delta_g', 
                                    'delta_nu', 'delta_num', 'delta_tot', 
                                    'delta_nonu', 'delta_tot_nodm', 'phi', 
                                    'v_cdm', 'v_b', 'v_b_cdm']
        self.dataset_names_pp = ['pk', 'pkchi', 'Trans']

    def init_cosmology(self, 
                       h=0.6735,
                       omc=0.2645,
                       omb=0.0493,
                       omk=0.0,
                       mnu=0.06,
                       nnu=3.044,
                       tau=0.0544,
                       ns=0.9649,
                       As=2.1e-9,
                       sigma8=0.8111,
                       nkpoints=1000,
                       minkh=5e-6,
                       maxkh=5e3,
                       redshift=0):
        """
        Args:
            minkh, maxkh (float): boundaries of CLASS/CAMB-calculated PS
            maxkh_extrap (float): maxkh, but for extrapolated part, that starts at maxkh
            extrap_fraction (float/int): 
                <1: fraction of original function to extrapolate with. 
                    i.e. 0.3 will extrapolate based on the last 30% of the minkh/maxkh interval 
                >1: number of points based on which to perform extrapolation.
                    i.e. 3 will extrapolate based on the last 3 points in the minkh/maxkh interval 
            highk_mode (str): which way highest k-values between maxkh and maxkh_extrap would be achieved. Values:
                "loglin_extrap" - linear extrapolation in log-log
                "analytic" - analytic expression for TF. For now, not implemented.
                "" - no highest-k modes will be achieved, the TF would cap out at maxkh
        """
        # Cosmology parameters
        self.h        = h 
        self.omc      = omc 
        self.omb      = omb 
        self.omk      = omk 
        self.mnu      = mnu 
        self.nnu      = nnu 
        self.tau      = tau 
        self.ns       = ns 
        self.As       = As 
        self.sigma8   = sigma8 
        # Output powerspectrum parameters
        self.nkpoints = nkpoints 
        self.minkh    = minkh 
        self.maxkh    = maxkh 
        self.redshift = redshift 
        # Derived parameters
        self.H0       = 100*self.h
        self.omch2    = self.omc * self.h**2 # Omega_cdm * h^2
        self.ombh2    = self.omb * self.h**2 # Omega_baryon * h^2

        self.handle_params_errors()

    def init_cosmology_from_run(self, run, Omega_k=0.0, nnu=3.044, nkpoints=1000, minkh=5e-6, maxkh=5e3, redshift=0):
        """
        Initialize cosmology from PeakPatch object
        Args:
            run (PeakPatch): object, corresponding to PeakPatch run.
            Omega_k (float): equivalent density fraction of curvature.
            nkpoints (int): number of points in the final power spectrum.
            minkh (float): minimum k value of final power spectrum.
            maxkh (float): maximum k value of final power spectrum.
            redshift (float): redshift of the po~r spectrum.
        """
        # Cosmology parameters
        self.init_cosmology(
        h        = run.h ,
        omc      = run.Omx ,
        omb      = run.OmB ,
        omk      = Omega_k,
        mnu      = run.mnu ,
        nnu      = nnu,
        tau      = run.tau ,
        ns       = run.ns ,
        As       = run.As ,
        sigma8   = run.sigma8 ,
        nkpoints = nkpoints ,
        minkh    = minkh ,
        maxkh    = maxkh ,
        redshift = redshift)

    # CLASS/CAMB calls
    
    def create_TF_CLASS(self):
        """
        Creates transfer functios using CLASS
    
        Returns:
            self (Transfer_data): TFs
        """
        
        # Set up the transfer data class
        Pk_renorm = 1 #(2 * np.pi * self.h)**3 # Renormalization constant to CAMB format
        Pk_renorm = self.h**(-3)
        z = self.redshift
    
        # Set up CLASS
        if self.output_type == 4:
            As_CLASS = self.As * 1e9
        else:
            As_CLASS = self.As
        LambdaCDM = Class()
        LambdaCDM.set({'omega_b':  self.ombh2, # Little omega, omega = Omega * h^2
                       'omega_cdm':self.omch2, # Little omega, omega = Omega * h^2
                       'h':        self.h,
                       'A_s':      As_CLASS,
                       'n_s':      self.ns,
                       'tau_reio': self.tau,
                       'N_ur':     self.nnu, # Eff number of massless nu species
                       'N_ncdm':   self.mnu, # Mass of massive neutrinos
                       'output':'mTk,vTk,tCl,pCl,lCl,mPk,dTk',
                       'lensing':'yes',
                       'k_min_tau0':self.minkh,
                       'z_pk': z,
                       'P_k_max_h/Mpc':self.maxkh,
                       #'gauge': 'newtonian',
                       })
        LambdaCDM.compute()
    
        # Sigma8 renormalization setup
        s8 = LambdaCDM.sigma8()
        print("sigma_8 pre normalization = ", s8)
        self.norm = (self.sigma8 / s8)**2  # Normalization constant
    
        # Power spectrum calculation (for PeakPatch)
        kk = np.logspace(np.log10(self.minkh),np.log10(self.maxkh),self.nkpoints) # k in h/Mpc
        if self.log: print(f"Number of points: {len(kk)}")
        h = LambdaCDM.h() # get reduced Hubble for conversions to 1/Mpc
        Pk = [] # P(k) in (Mpc/h)**3
        for k in kk:
            Pk.append(LambdaCDM.pk(k*h,z) / Pk_renorm ) # function .pk(k,z)
    
        # Obtaining transfer functions
        transfer_dict = LambdaCDM.get_transfer(z=z, output_format='camb')
        transfer_dict_v = LambdaCDM.get_transfer(z=z, output_format='class') # No velocity TFs in CAMB format
    
        self.kh        = transfer_dict['k (h/Mpc)']
        self.delta_cdm = transfer_dict['-T_cdm/k2']
        self.delta_b   = transfer_dict['-T_b/k2']
        self.delta_g   = transfer_dict['-T_g/k2']
        self.delta_nu  = transfer_dict['-T_ur/k2']
        self.delta_num = transfer_dict['-T_ncdm/k2']
        self.delta_tot = transfer_dict['-T_tot/k2']
    
        #Pk = self.delta_tot**2 * self.kh**self.ns
    
        self.v_b       = self.delta_b * h**2
        self.v_cdm     = self.delta_cdm * h**2
        self.v_b_cdm   = np.abs(self.v_b - self.v_cdm) # This one is not done correctly for some reason...
    
        self.phi         = transfer_dict_v['phi'] # Irrelevant for anything
        self.delta_nonu  = self.delta_tot - self.delta_nu - self.delta_num # Irrelevant for anything
        self.delta_tot_nodm = self.delta_tot # Irrelevant for anything
        if self.log: self.print_data_sizes()
    
        self = self.calc_pkp_ps_params(kk, Pk)
        #print("Dict:")
        #print(transfer_dict_v.keys())
        return self

    # Extrapolation

    def log_linear_extrap(self, x, *params):
        """
        Linear extrapolation function for extending the data for high x values
        """
        return params[0] * x + params[1]

    def extrap_params(self, 
                      maxkh_extrap=5e5,
                      minkh_extrap=1e-6,
                      extrap_fraction=0.05,
                      lowk_mode="loglin_extrap",
                      highk_mode="loglin_extrap"):
        """
        Set up the parameters for extrapolation of TF/PS.

        Args:
            maxkh_extrap (float): maxkh, but for extrapolated part, that starts at maxkh
            minkh_extrap (float): minkh, but for extrapolated part, that ends at minkh
            extrap_fraction (float/int): 
                <1: fraction of original function to extrapolate with. 
                    i.e. 0.3 will extrapolate based on the last 30% of the minkh/maxkh interval 
                >1: number of points based on which to perform extrapolation.
                    i.e. 3 will extrapolate based on the last 3 points in the minkh/maxkh interval 
            highk_mode (str): which way highest k-values between maxkh and maxkh_extrap would be achieved. Values:
                "loglin_extrap" - linear extrapolation in log-log
                "analytic" - analytic expression for TF. For now, not implemented.
                "" - no highest-k modes will be achieved, the TF would cap out at maxkh
            lowk_mode (str): same as highk_mode, but for low-k
        """
        # Cosmology parameters
        self.highk_mode = highk_mode
        self.lowk_mode = lowk_mode
        self.extrap_fraction = extrap_fraction
        self.maxkh_extrap = maxkh_extrap
        self.minkh_extrap = minkh_extrap
        self.possible_highk_modes = [ "loglin_extrap", "analytic", "" ]
        assert self.highk_mode in self.possible_highk_modes, f"highk_mode should be one of the following values: {self.possible_highk_modes}"
        assert self.lowk_mode  in self.possible_highk_modes, f"lowk_mode should be one of the following values: {self.possible_highk_modes}"

    def choose_extrapolation_scheme(self, extrapolation_func, min_k_interp, max_k_interp, extrap_points, max_mode=True):
        """
        Prepares for extrapolation, chooses extrapolation scheme, performs extrapolation.

        Args:
            extrapolation_func (func): specifies a function to use for extrapolation 
            min_k_interp (float): minimum k value used to fit to the linear relationship
            max_k_interp (float): maximum k value used to fit to the linear relationship
            extrap_points (int): number of points in the extrapolated region
            num_params (int): number of parameters in extrapolation_func to fit for
            max_mode (bool): whether to extend beyond min or max of interpolation region
        """
        if self.log: self.print_data_sizes()
        max_k_extrap = self.maxkh_extrap
        min_k_extrap = self.minkh_extrap
        x_data, dataset_names, _ = self.get_xdata()
        x_extrap = np.array([])

        if self.log: 
            print(f"max_k_interp: {max_k_interp}")
            print(f"max_k_extrap: {max_k_extrap}")
            print(f"min_k_interp: {min_k_interp}")
            print(f"min_k_extrap: {min_k_extrap}")
            print(f"extrap_points: {extrap_points}")
        if max_mode:
            extrap_bnd = max_k_extrap
            extrap_mode = self.highk_mode
        else:
            extrap_bnd = min_k_extrap
            extrap_mode = self.lowk_mode

        if extrap_mode == "loglin_extrap":
            x_extrap = self.power_law_extrapolation(extrapolation_func, dataset_names, x_data, min_k_interp, \
                                        max_k_interp, extrap_bnd, extrap_points, 2)
        elif extrap_mode == "analytic":
            if self.output_type == 4:
                x_extrap = self.analytic_tf_extrapolation_pp(x_data, max_k_interp, 
                                                             extrap_bnd, extrap_points)
            else:
                raise NotImplemented(f"Only PeakPatch output_type is currently supported for analytic extrapolation")
        elif extrap_mode == "":
            pass
        else:
            raise ValueError(f"Unknown extrapolation mode option: {extrap_mode}")

        if self.output_type == 4: # PeakPatch output format
            if max_mode: self.kk = np.append(x_data, x_extrap)
            else: self.kk = np.append(x_extrap, x_data)
            self.calculate_pkchi() # pkchi has to be recalculated
        else:
            if max_mode: self.kh = np.append(x_data, x_extrap)
            else: self.kh = np.append(x_extrap, x_data)
        if self.log: self.print_data_sizes()

    def power_law_extrapolation(self, extrapolation_func, dataset_names, x_data, min_k_interp, max_k_interp, extrap_bnd, extrap_points, num_params):
        """
        Extrapolates the transfer function,
        by fitting a line between k_interp points and extrapolating until 
        extrap_bnd.

        Args:
            extrapolation_func (func): specifies a function to use for extrapolation 
            dataset_names (list): list of names of attributes to extrapolate
            x_data (np.array): k-values
            min_k_interp (float): minimum k value used to fit to the linear relationship
            max_k_interp (float): maximum k value used to fit to the linear relationship
            extrap_points (int): number of points in the extrapolated region
            num_params (int): number of parameters in extrapolation_func to fit for
        """
        i=0
        p0 = [1.0] * num_params # initial guess
        if self.log:
            self.log: print(f"Starting extrapolation...")
            self.log: print(f"x: {x_data}")
            print(f"extrap_bnd: {extrap_bnd}")
            print(f"max_k_interp: {max_k_interp}")
            print(f"extrap_points: {extrap_points}")
        if extrap_bnd > max_k_interp:
            x_extrap = np.logspace(np.log10(max(x_data)), np.log10(extrap_bnd), extrap_points)
            x_extrap = x_extrap[1:]
            if self.log: print(f"Max extrapolation")
        else:
            x_extrap = np.logspace(np.log10(extrap_bnd), np.log10(min(x_data)), extrap_points)
            x_extrap = x_extrap[:-2]
            if self.log: print(f"Min extrapolation")
        if self.log: print(f"x_extrap: {x_extrap}")
        if self.log: print(f"min_k_interp: {min_k_interp}")
        for attr_name in dataset_names:
            y_data = getattr(self, attr_name)
            try:
                x_data_masked, y_data_masked = zip(
                        *[(x,y) for x,y in zip(x_data, y_data) 
                          if min_k_interp <= x <= max_k_interp])
            except:
                raise ValueError(f"Too few points in interpolation region. Something is wrong...")
            x_data_log = np.log10(x_data_masked)
            y_data_log = np.log10(np.abs(y_data_masked + np.full(len(y_data_masked), self.epsilon)) )
            if self.log: 
                print(f"y_data id: {i}"); i+=1
                print(f"len(x_data_log): {len(x_data_log)}")
                print(f"len(y_data): {len(y_data)}")
                print(f"len(y_data_log): {len(y_data_log)}")
            popt, pcov = curve_fit(extrapolation_func, x_data_log, y_data_log, p0=p0)
            y_extrap = extrapolation_func(np.log10(x_extrap), *popt)
            y_extrap = 10**y_extrap
            if extrap_bnd > max_k_interp:
                setattr(self, attr_name, np.append(y_data, y_extrap))
            else:
                setattr(self, attr_name, np.append(y_extrap, y_data))
        return x_extrap
    
    def high_k_extrapolation(self):
        """
        Performs high-k extrapolation with all available methods
        """
        maxkh_log = np.log10(self.maxkh)
        minkh_log = np.log10(self.minkh)
        maxkh_extrap_log = np.log10(self.maxkh_extrap)
        points_density = ( maxkh_log - minkh_log ) / self.nkpoints
        extrap_points = int( ( maxkh_extrap_log - maxkh_log ) / points_density )
        if self.extrap_fraction < 1: # Extrapolation based on fraction of total interval
            inner_interp_bnd = (1 - self.extrap_fraction) * maxkh_log + \
                    self.extrap_fraction * minkh_log
        elif self.extrap_fraction > 1: # Extrapolation based on number of points
            #dklog = (maxkh_log - minkh_log) / (self.nkpoints - 1)
            #dk_extrap = dklog * int(self.extrap_fraction + 1)
            #inner_interp_bnd = maxkh_log - dk_extrap
            x_data, _, _ = self.get_xdata()
            inner_interp_bnd = np.log10(x_data[-1 - int(self.extrap_fraction)])
            inner_interp_bnd *= (1 - 1e-3)
        else:
            raise ValueError(f"Cannot have extrap_fraction values between 1 and 2.")
        self.choose_extrapolation_scheme(self.log_linear_extrap, 10**inner_interp_bnd, \
                                         self.maxkh, extrap_points, max_mode=True)

    def low_k_extrapolation(self):
        """
        Performs low-k extrapolation with all available methods
        """
        maxkh_log = np.log10(self.maxkh)
        minkh_log = np.log10(self.minkh)
        minkh_extrap_log = np.log10(self.minkh_extrap)
        points_density = ( maxkh_log - minkh_log ) / self.nkpoints
        extrap_points = - int( ( minkh_extrap_log - minkh_log ) / points_density )
        if self.extrap_fraction < 1: # Extrapolation based on fraction of total interval
            extrap_fraction = 1 - self.extrap_fraction
            inner_interp_bnd = (1 - self.extrap_fraction) * maxkh_log + \
                    self.extrap_fraction * minkh_log
        elif int(self.extrap_fraction) > 1: # Extrapolation based on number of points
            #dklog = (maxkh_log - minkh_log) / self.nkpoints
            #dk_extrap = dklog * int(self.extrap_fraction + 1)
            #inner_interp_bnd = minkh_log + dk_extrap
            x_data, _, _ = self.get_xdata()
            inner_interp_bnd = np.log10(x_data[int(self.extrap_fraction) + 1])
            inner_interp_bnd *= (1 + 1e-3)
        else:
            raise ValueError(f"Cannot have extrap_fraction values between 1 and 2.")
        self.choose_extrapolation_scheme(self.log_linear_extrap, \
                                         self.minkh, \
                                         10**inner_interp_bnd, \
                                         extrap_points, max_mode=False)

    # Error correction

    def replace_invalid_values(self, array, replacement_value):
        """
        Replaces nans and infs with another value
        """
        array = np.array(array)  # Ensure it's a NumPy array
        invalid_mask = np.isnan(array) | np.isinf(array)  # Mask for NaN or Inf
        array[invalid_mask] = replacement_value  # Replace invalid values
        return array

    def drop_n_points_from_all_data(self, n):
        """Drops n points from all datasets"""
        x_data, dataset_names, x_attr_name = self.get_xdata()
        setattr(self, x_attr_name, x_data[n:])
        for attr_name in dataset_names:
            y_data = getattr(self, attr_name)
            setattr(self, attr_name, y_data[n:])
    
    # Assertions

    def handle_0_value_error(self, varname, var):
        """Checks if var == 0. Exits if that's the case"""
        assert var !=0, print(f"{varname} was set to 0. This breaks the code. Please, fix it.")

    def handle_params_errors(self):
        self.handle_0_value_error("h", self.h)
        self.handle_0_value_error("As", self.As)
        self.handle_0_value_error("tau", self.tau)
        self.handle_0_value_error("sigma8", self.sigma8)
        self.handle_0_value_error("minkh", self.minkh)
        self.handle_0_value_error("minkh", self.minkh)
        self.handle_0_value_error("nkpoints", self.nkpoints)

    # Logging

    def print_data_sizes(self):
        """
        Prints the sizes of data
        """
        print(f"self.kh size: {self.kh.size}")
        print(f"delta_cdm size: {self.delta_cdm.size}")
        print(f"delta_b size: {self.delta_b.size}")
        print(f"delta_g size: {self.delta_g.size}")
        print(f"delta_nu size: {self.delta_nu.size}")
        print(f"delta_num size: {self.delta_num.size}")
        print(f"delta_tot size: {self.delta_tot.size}")
        print(f"delta_nonu size: {self.delta_nonu.size}")
        print(f"delta_tot_nodm size: {self.delta_tot_nodm.size}")
        print(f"phi size: {self.phi.size}")
        print(f"v_cdm size: {self.v_cdm.size}")
        print(f"v_b size: {self.v_b.size}")
        print(f"v_b_cdm size: {self.v_b_cdm.size}")
        print(f"max kh: {np.max(self.kh)}")
        print(f"self.kk size: {self.kk.size}")
        print(f"self.Trans size: {self.Trans.size}")
        print(f"self.pkchi size: {self.pkchi.size}")

    def print_comology_params(self):
        """Prints cosmological/run parameters"""
        print(f"h:     {self.h}")
        print(f"omc:   {self.omc}")
        print(f"omb:   {self.omb}")
        print(f"omk:   {self.omk}")
        print(f"mnu:   {self.mnu}")
        print(f"nnu:   {self.nnu}")
        print(f"tau:   {self.tau}")
        print(f"ns:    {self.ns}")
        print(f"As:    {self.As}")
        print(f"s8:    {self.sigma8}")
        print(f"nkpts: {self.nkpoints}")
        print(f"minkh: {self.minkh}")
        print(f"maxkh: {self.maxkh}")
        print(f"z_0:   {self.redshift}")

    # Analytic TF 

    def analytic_tf_extrapolation_pp(self, x_data, max_k_interp, max_k_extrap, extrap_points):
        """
        Extrapolates the transfer function,
        by using analytic approximation from eq. 8.69, Dodelson 2003

        Args:
            x_data (np.array): k-values
            max_k_interp (float): maximum k value used to fit to the linear relationship
            extrap_points (int): number of points in the extrapolated region
        """
        i=0
        x_extrap = np.logspace(np.log10(max_k_interp), np.log10(max_k_extrap), extrap_points)
        y_data = self.Trans # We're going to extrapolate the transfer function only
        y_extrap = self.analytic_tf(x_extrap)
        setattr(self, 'Trans', np.append(y_data, y_extrap))
        self.kk = np.append(x_data, x_extrap)
        pkzeta = self.make_pkzeta()
        self.pk = self.Trans**2 * pkzeta
        return x_extrap

    def analytic_tf(self, k):
        """
        Analytic transfer function at value k 
        """
        self.omr = 2.47 * 10**(-5)
        omM_tot = self.omc + self.omb # Mass of baryons + CDM
        #z_eq = 3400
        #a_eq = 1 / (1 + z_eq)
        a_eq = self.omr / omM_tot
        mpc_to_km = 3.0857 * 10**19
        c = 3 * 10**5 # in km/s
        k_eq = self.H0 / c * np.sqrt(2 * omM_tot / a_eq) # in c=1 units
        #k_eq = a_eq * H_eq
        Tf = 12 * k_eq**2 / k**2 * np.log(0.12 * k / k_eq)
        #A = 6.4
        #B = 0.4
        #Tf = 15/4 * self.omc * self.H0**2 / k**2 / self.a_eq * A * np.log((4 * B * np.e**(-3) np.sqrt(2) * k)/k_eq)
        if self.log: print(f"a_eq = {a_eq}")
        if self.log: print(f"k_eq = {k_eq}")
        #if self.log: print(f"H_eq = {H_eq}")
        return Tf*10**11

    # PeakPatch calculations

    def calc_pkp_ps_params(self, kh, pk):
        """
        Calculates PeakPatch-related PS details
    
        Args:
            params (Run_params): cosmology and power spectrum parameters
            kh (np array): Wavenumbers
            pk (np array): Power spectrum
            norm (float): Power spectrum normalization constant
    
        Returns:
            Trans (np array): PeakPatch Transfer function
            pkchi (np array): PeakPatch chi power spectrum
        """
        pk = self.norm * np.array(pk) / (2. * np.pi * self.h)**3   # Normalized P_m(z=0,k)
        self.kk = kh
    
        # Get transfer function
        pkzeta = self.make_pkzeta()
        Trans = np.sqrt(pk / pkzeta)
    
        self.pk = pk
        self.Trans = Trans
        self.calculate_pkchi()
        return self

    def make_pkzeta(self):
        """
        Creates primordial zeta power spectrum from self.kk
        """
        # Primordial zeta power spectrum
        ko = 0.05
        pkzeta = 2 * np.pi**2 * self.As / self.kk**3 * (self.kk / ko)**(self.ns - 1)
        return pkzeta

    def calculate_pkchi(self):
        """
        Calculates pkchi (light field power spectra) from kh        
        """
        k = self.kk * self.h               # Wavenumber k in Mpc^-1
        # Light field power spectra
        Achi = (5.e-7)**2
        pkchi = 2 * np.pi**2 * Achi / k**3  # In units of sigmas
        pkchi = pkchi / (2 * np.pi)**3  # For pp power spectra
        self.pkchi = pkchi

    def renormalize_transfer(self):
        """
        renormalizes transfer functions
    
        Returns:
            self (Transfer_data): renormalized TFs
        """
        sqrt_norm = np.sqrt(self.norm)
        self.delta_cdm = self.delta_cdm * sqrt_norm 
        self.delta_b = self.delta_b * sqrt_norm 
        self.delta_g = self.delta_g * sqrt_norm 
        self.delta_nu = self.delta_nu * sqrt_norm 
        self.delta_num = self.delta_num * sqrt_norm 
        self.delta_tot = self.delta_tot * sqrt_norm 
        self.delta_nonu = self.delta_nonu * sqrt_norm 
        self.delta_tot_nodm = self.delta_tot_nodm * sqrt_norm 
        self.phi = self.phi * sqrt_norm 
        self.v_cdm = self.v_cdm * sqrt_norm 
        self.v_b = self.v_b * sqrt_norm 
        self.v_b_cdm = self.v_b_cdm * sqrt_norm 
    
        return self

    # Outputs - data

    def create_and_save_TF(self, output_file, TF_src):
        """
        This function is fully responsible for creating and saving transfer functions of a certain type, from start to finish
    
        Args:
            output_file (str): name of the output table file
            TF_src (str): what code will create the transfer functions (CAMB or CLASS)
    
        Returns:
            data (np array): a table with all the transfer functions in output format
        """
        print("Starting to generate TFs/PS...")
        start = time.time()
        if TF_src == 'CAMB':
            self.self = self.create_TF_CAMB()
            self.self = self.renormalize_transfer()
            print("Finished calculating CAMB TF")
        elif TF_src == 'CLASS':
            self = self.create_TF_CLASS()
            self = self.renormalize_transfer()
            print("Finished calculating CLASS TF")
        else:
            print(f"Wrong TF_src variable value: {TF_src}. Allowed values are: CAMB, CLASS")
            exit(1)
        end = time.time()
        elapsed = end - start
        print(f"Generating TF/PS dataset took {elapsed:.2f} sec")
        if self.highk_mode != "" and self.maxkh < self.maxkh_extrap:
            print(f"Performing high-k extrapolation with {self.highk_mode} method...")
            self.high_k_extrapolation()
        # Need to drop the 1st datapoint for some reason - not sure why it is messed up
        self.drop_n_points_from_all_data(1)
        if self.lowk_mode != "" and self.minkh > self.minkh_extrap:
            print(f"Performing low-k extrapolation with {self.lowk_mode} method...")
            self.low_k_extrapolation()
        print(f"max kh: {np.max(self.kh)}")
        header, data = self.get_formatted_data()
        if self.Save_output(header, data, output_file):
            print("Error on saving output.")
            exit(1)
        return data

    def create_and_save_TF_class(self, output_file):
        """
        This function is fully responsible for creating and saving transfer functions of a certain type, from start to finish, using CLASS
    
        Args:
            output_file (str): name of the output table file
    
        Returns:
            data (np array): a table with all the transfer functions in output format
        """
        return self.create_and_save_TF(output_file, 'CLASS')

    def create_and_save_TF_camb(self, output_file):
        """
        This function is fully responsible for creating and saving transfer functions 
        of a certain type, from start to finish, using CAMB
    
        Args:
            output_file (str): name of the output table file
    
        Returns:
            data (np array): a table with all the transfer functions in output format
        """
        return self.create_and_save_TF(output_file, 'CAMB')

    # Outputs - other
    
    def get_formatted_data(self):
        """
        Formats the data in the format that is requested
    
        Returns:
            header (str): header for output table
            data (np array): a table with all the transfer functions in output format
        """
        if self.output_type == 2: # Debug format
            header = 'k, delta_cdm'
            data = np.vstack([self.kh * self.h,
                              self.delta_cdm
                            ]).T
            return header, data
        elif self.output_type == 4: # PeakPatch format
            header = ''
            data = np.vstack([self.kk * self.h,
                              self.pk,
                              self.Trans,
                              self.pkchi
                            ]).T
            return header, data
        elif self.output_type == 7: # Legacy format (usage unknown)
            header = 'k/h Delta_CDM/k2 Delta_b/k2 Delta_g/k2 Delta_nu/k2 Delta_tot/k2 Phi'
            data = np.vstack([self.kh,
                              self.delta_cdm/k**2,
                              self.delta_b/k**2,
                              self.delta_g/k**2,
                              self.delta_nu/k**2,
                              self.delta_tot/k**2,
                              self.phi
                            ]).T
            return header, data
        elif self.output_type == 13: # MUSIC format
            header = 'k, delta_cdm, delta_b, delta_g, delta_nu, delta_num, delta_tot, delta_nonu, delta_tot_nodm, phi, v_cdm, v_b, v_b_cdm'
            data = np.vstack([self.kh * self.h,
                              self.delta_cdm,
                              self.delta_b,
                              self.delta_g,
                              self.delta_nu,
                              self.delta_num,
                              self.delta_tot,
                              self.delta_nonu,
                              self.delta_tot_nodm,
                              self.phi,
                              self.v_cdm,
                              self.v_b,
                              self.v_b_cdm
                            ]).T
            return header, data
        
        print(f"Unsupported output type: {self.output_type}")
        print(f"Supported types: 2, 4, 7, 13")
        exit(1)
    
    def Save_output(self, header, data, output_file):
        """
        Save the output table
    
        Args:
            header (str): header for output table 
            data (np array): a table with all the transfer functions in output format
            output_file (str): name of the output table file
    
        Returns:
            int: 0 on success
        """
        if header == '' and output_file is not None:
            np.savetxt(output_file, data, fmt='%0.8e')
        elif output_file is not None:
            np.savetxt(output_file, data, header=header, fmt='%0.8e')
        if output_file is not None:
            print(f"Data saved to {output_file}")
        else:
            print("No output file was saved")
        return 0
    
    def diagnostic_plot(self, datasets, labels, figure_path):
        """
        A function that plots TFs for diagnostic purposes

        Args:
            datasets (list of np arrays): list with tables with all transfer functions
            labels (list of str): labels on the plots.
            figure_path (str): name of the output figure.

        Returns:
            int: 0 if succes, 1 if fail
        """
        if self.output_type !=2:
            print(f"For diagnostic plot, please set output_type to 2.")
            print(f"Your value: {self.output_type}")
            return 1
        if len(labels) != len(datasets):
            print(f"labels and datasets have a different size.")
            print(f"Size of labels: {len(labels)}, size of datasets: {len(datasets)}")
            return 1

        for i in range(len(datasets)):
            plt.plot(datasets[i][:,0],  datasets[i][:,1], label=labels[i])
        plt.xlabel('k')
        plt.ylabel('P(k)')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(figure_path)
        return 0

    # Utilities

    def set_output_type(self, output_type):
        """
        Change output type.
        Args:
            output_type (int): a type of output requested: 
                2 for Debug
                4 for PeakPatch 
                7 for Legacy
                13 for MUSIC 
        """
        # Cosmology parameters
        self.output_type = output_type

        if output_type not in self.allowed_output_types:
            print(f"output_type {output_type} is not allowed")
            print(f"possible values: {self.allowed_output_types}")
            exit(1)
    
    def get_xdata(self):
        """Obtains data for the x dataset (kk or kh), and dataset names,
        based solely on current set output type"""
        if self.output_type == 4: # PeakPatch output format
            dataset_names = self.dataset_names_pp
            x_attr_name = 'kk'
        else:
            dataset_names = self.dataset_names_music
            x_attr_name = 'kh'
        x_data = getattr(self, x_attr_name)
        return x_data, dataset_names, x_attr_name


if __name__ == "__main__":

    log = False
    class_file_pp = 'CLASS_pp.dat'
    class_file_music = 'CLASS_music.dat'
    camb_file_pp = 'CAMB_pp.dat'
    camb_file_music = 'CAMB_music.dat'
    figure_path = 'CLASS_vs_CAMB.png'

    params_file = './param/parameters.ini'
    #extrapolation_scheme = "analytic"
    extrapolation_scheme = "loglin_extrap"
    run_dir = '.'
    debug_pptools=False
    minkh = 1e-5
    maxkh = 1e2
    maxkh_extrap = 1e2
    minkh_extrap = 1e-5
    extrap_fraction = 2 # Last 2 points

    # Create PeakPatch object
    TFCalc = calculate_TF(13, log=log)

    # Variant 1: import cosmology from a run
    #run = PeakPatch(params_file=params_file, run_dir=run_dir, debug=debug_pptools)
    #TFCalc.init_cosmology_from_run(run, minkh=minkh, maxkh=maxkh)

    # Variant 2: set up your own cosmology (or use defaults)
    TFCalc.init_cosmology(h = 0.6735,
                          omc = 0.2645,
                          omb = 0.0493,
                          omk = 0.0,
                          mnu = 0.06,
                          nnu = 3.044,
                          tau = 0.0544,
                          ns = 0.9649,
                          As = 2.1e-09,
                          sigma8 = 0.8111,
                          nkpoints = 1000,
                          minkh = minkh,
                          maxkh = maxkh,
                          redshift = 0)

    TFCalc.extrap_params(maxkh_extrap = maxkh_extrap,
                         minkh_extrap = minkh_extrap,
                         extrap_fraction = extrap_fraction,
                         highk_mode = extrapolation_scheme,
                         lowk_mode = extrapolation_scheme)

    TFCalc.print_comology_params()
    data = TFCalc.create_and_save_TF_class(None)

    plt.plot(data[:, 0], data[:, 1])
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('out.png')
