"""

    Functions which generate the random variables used to generate the residual

"""

import numpy as np

import src.constants as const
from scipy.stats import maxwell
from scipy import integrate
from scipy.interpolate import interp1d

def gen_dhats(num_pulsar):
    """

        Returns a list of vectors giving the positions of the pulsars on the sphere

    """

    pulsar_theta = np.arccos(1-2*np.random.rand(num_pulsar))
    pulsar_phi = 2*np.pi*np.random.rand(num_pulsar)
    
    # d_hat for each pulsar (3,num_pulsar)
    d_hat = np.zeros((3,num_pulsar))
    d_hat[0] = np.sin(pulsar_theta)*np.cos(pulsar_phi)
    d_hat[1] = np.sin(pulsar_theta)*np.sin(pulsar_phi)
    d_hat[2] = np.cos(pulsar_theta)

    return d_hat.T # (N, 3)

def gen_positions(max_R, num_object):
    """

        Generates a list of positions
        
    """

    position_r = max_R*np.cbrt(np.random.rand(num_object))      #r = u^(1/3) where u~rand[0,1), kpc
    position_theta = np.arccos(1-2*np.random.rand(num_object))  #cos(theta) ~ rand[-1,1)
    position_phi = 2*np.pi*np.random.rand(num_object)           #phi ~ rand[0,2pi)

    position = np.zeros((3, num_object))
    position[0] = position_r*np.sin(position_theta)*np.cos(position_phi)
    position[1] = position_r*np.sin(position_theta)*np.sin(position_phi)
    position[2] = position_r*np.cos(position_theta)

    return position.T # (N, 3)

def gen_velocities(v_0, v_Esc, v_E, num_object):
    """
        
        Generates a list of velocities

    """

    cdf_v_Esc = maxwell.cdf(v_Esc, scale=v_0/np.sqrt(2))

    velocity_r = maxwell.ppf(np.random.rand(num_object)*cdf_v_Esc, scale=v_0/np.sqrt(2)) #kpc/yr
    velocity_theta = np.arccos(1-2*np.random.rand(num_object))
    velocity_phi = 2*np.pi*np.random.rand(num_object)
                
    velocity = np.zeros((3, num_object))
    velocity[0] = velocity_r*np.sin(velocity_theta)*np.cos(velocity_phi)
    velocity[1] = velocity_r*np.sin(velocity_theta)*np.sin(velocity_phi)
    velocity[2] = velocity_r*np.cos(velocity_theta)

    return velocity.T # (N, 3)

def gen_masses(num_objects, use_HMF = False, log10_M = -6, HMF_path = '', log10_M_min = -12):
    """

        Generates a list of masses

    """

    if use_HMF == False:

        mass = np.ones(num_objects)*10**(log10_M)

    else:

        inv_cdf = mass_dist(HMF_path, 10**log10_M_min)[0]
        mass = inv_cdf(np.random.rand(num_objects))

    return mass # (N)

def set_num_objects(max_R, log10_f = 0, log10_M = -6, use_HMF = False, HMF_path='', 
        log10_M_min = -12, min_num_object = 1, verbose = False):
    """
        Sets the number of subhalos in the simulation and the simulation radius
    """

    volume = (4*np.pi/3)*(max_R**3)
    
    if not use_HMF:

        num_density = (10**log10_f)*(const.rho_DM)/(10**log10_M)
        final_m_min = log10_M_min 

        if int(volume*num_density) < min_num_object:

            num_object = min_num_object

            if verbose == True:
                
                print('!!! Warning !!!')
                print('    Physical number of subhalos, n x V is less than min_num_object.')
                print('    Setting the number of subhalos to '+str(num_object))
                print()

        else:

            num_object = int(volume*num_density)

    else:

        m_min_min_num_object = get_M_min(HMF_path, min_num_object/(10**log10_f)/volume)
        
        # take the smaller one between log10_M_min and m_min_min_num_object
        final_m_min = min(m_min_min_num_object, 10**log10_M_min)
    
        # number density of halos (kpc^-3)
        num_density = 10**log10_f*mass_dist(HMF_path, final_m_min)[1]
        num_object = int(volume*num_density)

    # calculate the new volume using the final number of halos
    volume = num_object/num_density # kpc^3
    max_R = (3*volume/(4*np.pi)) ** (1/3) # kpc

    return [num_object, max_R, final_m_min]


def mass_dist(HMF_path, m_min):
    
    # return the CDF of the mass distribution and the number density of subhalos
    
    # HMF_path: path with the HMF file:
        # first column: subhalo mass (solar mass)
        # second column: dn/dlog10M (pc^-3)
    # m_min: minimum cut-off mass (solar mass)

    # return: 
        # inv_cdf: function that takes in a value from [0,1] and output mass M such that CDF(M) = the value
        # num_density_halo: number density of halos (kpc^-3)
    
    # read the halo mass function
    mass_raw = np.loadtxt(HMF_path, delimiter=',')[:,0] # solar mass
    hmf_raw = np.loadtxt(HMF_path, delimiter=',')[:,1] # pc^-3

    # cut off halos at low mass
    mass = mass_raw[mass_raw >= m_min] # solar mass
    hmf = hmf_raw[mass_raw >= m_min] * (1e9) # kpc^-3
    
    # number density of halos (kpc^-3) with mass < M
    num_density_halo_M = np.zeros(len(mass))
    for i in range(len(mass)):
        num_density_halo_M[i] = integrate.trapz(hmf[:i + 1], x = np.log10(mass[:i + 1]))
    
    # function for the inverse CDF of the HMF
    inv_cdf = interp1d(num_density_halo_M / num_density_halo_M[-1], mass)
    
    return inv_cdf, int(num_density_halo_M[-1])

def get_M_min(HMF_path, num_density):
    
    # return the M_min for a given halo number density
    # in other words, find x such that \int_x^\infty dn / dM dM = number density
    
    # HMF_path: path with the HMF file:
        # first column: subhalo mass (solar mass)
        # second column: dn/dlog10M (pc^-3)
    # num_density: halo number density (kpc^-3)

    # return: 
        # M_min
        # num_density_halo: number density of halos (kpc^-3)
        
    # convert the halo number density to pc^-3
    num_density_pc = num_density / (1e9) # pc^-3 
    
    # read the halo mass function, from high mass to low mass
    mass = np.flip(np.loadtxt(HMF_path, delimiter=',')[:,0]) # solar mass
    hmf = np.flip(np.loadtxt(HMF_path, delimiter=',')[:,1]) # pc^-3
    
    # number density of halos (pc^-3) with M < mass < M_max
    num_density_halo_M = np.zeros(len(mass))
    for i in range(len(mass)):
        num_density_halo_M[i] = np.abs(integrate.trapz(hmf[:i + 1], x = np.log10(mass[:i + 1])))
    
    # return error if the entire HMF is not sufficient to give the required number density
    if num_density_halo_M[-1] < num_density_pc:
        raise ValueError('min_num_object is too large. The entire HMF is not sufficient to give the required number density. Consider increasing the r-factor or decreasing min_num_object.')
    else:
        # find the first M (from high M) such that n within M < mass < M_max is greater than num_density_pc
        return mass[np.argmax(num_density_halo_M > num_density_pc)]





