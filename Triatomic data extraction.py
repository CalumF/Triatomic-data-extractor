# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 20:29:24 2019

@author: Calum
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def get_file_list():
    """ Returns file names as a list, given input of the folder name (if in cwd) or absolute
    folder path."""
    global data_folder
    data_folder = [input("Enter data folder:")]
    if data_folder[0][:len(os.path.expanduser('~'))] == os.path.expanduser('~'):
        data_folder.append(1)
        return os.listdir(data_folder[0])
    else:
        foldpath = os.path.join(os.getcwd(), data_folder[0])
        return os.listdir(foldpath)

def getline(file, lines):
    return [x for i, x in enumerate(file) if i in lines]

def data_extraction(files, fold):
    """ Arguments: files- file names to extract from as a list 
                   fold- the full folder path
    Extracts specifically located data, with tolerance for different line indexes.
    Returns bond length list, angle list, energy list"""
    allr = []
    allang = []
    allE = []
    if len(fold) > 1:
        fold_name = fold[0]
    else:
        fold_name = os.path.join(os.getcwd(), fold[0])
    for i in files:
        with open(os.path.join(fold_name, i), "r") as f:
            data_strings = getline(f, [112,167])
            if data_strings[1][:9] == " SCF Done" and data_strings[0][:9] == "     3  H":
                r = float(data_strings[0][13:21])
                HH_dist = float(data_strings[0][24:32])
                en = float(data_strings[1][22:36])
                allr.append(r)
                allang.append(np.degrees(2 * np.arcsin((HH_dist / (2 * r)))))
                allE.append(en)
            else:
                check = 0
                for j in f:
                    if j[:9] == " SCF Done":
                        check =+ 1
                        en = float(j[22:36])
                        allE.append(en)
                if check == 1:
                    r = float(data_strings[0][13:21])
                    HH_dist = float(data_strings[0][24:32])
                    allr.append(r)
                    allang.append(np.degrees(2 * np.arcsin((HH_dist / (2 * r)))))
    if len(allr) == len(allang) == len(allE):
        print('extract success')
    else:
        print('allr' + str(len(allr)))
        print('allang' + str(len(allang)))
        print('allE' + str(len(allE)))
    return allr, allang, allE

def plot_fxn(x, y, z):
    """ Plots a 3D graph of the lower 70% of the data by energy"""
    z_sort, x_sort, y_sort = [i for i in zip(*sorted(zip(z, x, y)))]
    scale = round(len(z_sort) * 0.6)
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x_sort[:scale], y_sort[:scale], z_sort[:scale], cmap = plt.cm.GnBu)
    ax.set_zlabel('Energy (Hartrees)')
    ax.set_ylabel('Bond Angle (degrees)')
    ax.set_xlabel('Bond Length (Å)')
    ax.view_init(43, 235)
    fig_name = input('To save plot, enter figure name and extension, or noplot to not save:')
    if fig_name != 'noplot':
        plt.savefig(fig_name, bbox_inches = 'tight')

def diff_wrt_equilibrium(bond_len, ang, energy):
    displacements = [i- bond_len[energy.index(min(energy))] for i in bond_len]
    angle_offset = [i- ang[energy.index(min(energy))] for i in ang]
    return displacements, angle_offset

def print_eq_geometry(bond_len, ang, energy):
    eq_bl = bond_len[energy.index(min(energy))]
    eq_ang = ang[energy.index(min(energy))]
    eq_eng = min(energy)
    print('\n' , 'The equilibrium geometry is:')
    print('Bond length: {:.2f} angstroms'.format(eq_bl))
    print('Angle      : {:.1f} degrees'.format(eq_ang))
    print('Energy     : {:.3f} hartrees'.format(eq_eng))

def fit_r_at_eqang(bond_len, ang, energy):
    displacement_eqang = []
    for idx, val in enumerate(bond_len):
        if -0.5 < ang[idx] < 0.5 and -0.11 < val < 0.11:
            displacement_eqang.append(val * 10**-10)
    energies_at_eqang = []
    for idx, val in enumerate(energy):
        if -0.5 < ang[idx] < 0.5 and -0.11 < bond_len[idx] < 0.11:
            energies_at_eqang.append(val * 4.35974 * 10**-18)

def bond_length_fit(displacement, angle_offset, energy):
    """ Fits the displacements vs energy at the equilibrium angle. Tries fits
    for increasing no. of data points from eq until a max least squares fit 
    error (RSS) is hit.
    Input is in units: angstroms, degrees, hartrees."""
    displacement_eqang = []
    for idx, val in enumerate(displacement):
        if -0.5 < angle_offset[idx] < 0.5:
            displacement_eqang.append(val * 10**-10)
    energies_at_eqang = []
    for idx, val in enumerate(energy):
        if -0.5 < angle_offset[idx] < 0.5:
            energies_at_eqang.append(val * 4.35974 * 10**-18)
    fit_r = [0]
    energy_min_idx = energies_at_eqang.index(min(energies_at_eqang))
    fit_energies = [min(energies_at_eqang)]
    for i in range(1, round(len(displacement_eqang)/2)):
        fit_r.append(displacement_eqang[energy_min_idx + i])
        fit_r.insert(0, displacement_eqang[energy_min_idx - i])
        fit_energies.append(energies_at_eqang[energy_min_idx + i])
        fit_energies.insert(0, energies_at_eqang[energy_min_idx - i])
        fit = np.polyfit(fit_r, fit_energies, 2, full=True)
        if fit[1].size > 0:
            if fit[1] > 1.5*10**-40:
                return quad_coeff
            else:
                quad_coeff = fit[0][0]
        else:
            quad_coeff = fit[0][0]


def angle_fit(displacement, angle_offset, energy):
    """ Fits the angular displacements vs energy at the equilibrium angle. Tries fits
    for increasing no. of data points from eq until a max least squares fit 
    error (RSS) is hit.
    Input is in units: angstroms, degrees, hartrees.
    """
    ang_at_eq_bond_length = []
    for idx, val in enumerate(angle_offset):
        if -0.05 < displacement[idx] < 0.05:
            ang_at_eq_bond_length.append(np.radians(val))
    energies_at_eq_bond_length = []
    for idx, val in enumerate(energy):
        if -0.05 < displacement[idx] < 0.05:
            energies_at_eq_bond_length.append(val * 4.35974 * 10**-18)
    fit_ang = [0]
    energy_min_idx = energies_at_eq_bond_length.index(min(energies_at_eq_bond_length))
    fit_energies = [min(energies_at_eq_bond_length)]
    for i in range(1, round(len(ang_at_eq_bond_length)/2)):
        fit_ang.append(ang_at_eq_bond_length[energy_min_idx + i])
        fit_ang.insert(0, ang_at_eq_bond_length[energy_min_idx - i])
        fit_energies.append(energies_at_eq_bond_length[energy_min_idx + i])
        fit_energies.insert(0, energies_at_eq_bond_length[energy_min_idx - i])
        fit = np.polyfit(fit_ang, fit_energies, 2, full=True)
        if fit[1].size > 0:
            if fit[1] > 1.5*10**-40:
                return quad_coeff
            else:
                quad_coeff = fit[0][0]
        else:
            quad_coeff = fit[0][0]

def calc_stretch(kr):
    f = (1/(2 * np.pi)) * np.sqrt(kr / (2 * 1.660539040 * 10 ** -27))
    f = f*33.35641*10**-12
    print('v1 = {:.2f} cm-1'.format(f))
    
def calc_bend(kr, eq_bond_length):
    f = (1/(2 * np.pi)) * np.sqrt(ktheta / ((eq_bond_length ** 2) * 0.5 * 1.660539040 * 10 ** -27))
    f = f*33.35641*10**-12
    print('v2 = {:.2f} cm-1'.format(f))    
    
    
        
    

bond_lengths, angles, energies = data_extraction(get_file_list(), data_folder)
plot_fxn(bond_lengths, angles, energies)
displacement_eq, angle_offset_eq = diff_wrt_equilibrium(bond_lengths, angles, energies)
kr = 2 * bond_length_fit(displacement_eq, angle_offset_eq, energies)
print('\n', 'The normal mode frequencies of the triatiomic in cm-1 are: ')
calc_stretch(kr)
ktheta = 2 * angle_fit(displacement_eq, angle_offset_eq, energies)
eq_r = bond_lengths[energies.index(min(energies))] * 10** -10
calc_bend(ktheta, eq_r)
print_eq_geometry(bond_lengths, angles, energies)