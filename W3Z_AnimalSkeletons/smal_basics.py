import json
import os
import pickle as pkl
import numpy as np
import pickle as pkl
from huggingface_hub import hf_hub_download

import sys, os

def align_smal_template_to_symmetry_axis(animal_type, v_template):

    # Define HuggingFace repo and paths# Define HuggingFace repo and paths
    repo_id = "WatermelonHCMUT/AnimalSkeletons"
    symmetry_paths = {
        'equidae': 'symmetry_indexes_horse.pkl',
        'canidae': 'symmetry_inds.json'  # Using same symmetry for now
    }

    if animal_type not in symmetry_paths:
        raise ValueError(f"Unsupported animal type: {animal_type}")

    # Download symmetry file to temp directory
    sym_path = hf_hub_download(
        repo_id=repo_id,
        filename=symmetry_paths[animal_type],
        local_dir="temp"
    )

    # These are the indexes of the points that are on the symmetry axis
    I = []
    if animal_type == 'equidae':
        I = [522,523,524,531,532,533,534,535,536,537,544,545,546,547,558,559,560,561,
         562,563,564,565,566,567,568,569,570,571,573,574,575,576,577,578,579,580,
         581,582,583,584,585,586,587,588,589,590,591,592,593,594,595,596,597,598,
         599,600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,639,
         640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,659,
         660,688,689,690,691,692,693,694,695,717,718,719,720,721,728,732,736,740,
         741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,758,
         759,760,762]
    elif animal_type == 'canidae':
        I = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 37, 55, 119, 120, 163, 209, 210, 211, 213, 216, 227, 326, 395, 452, 578, 910, 959, 964, 975, 976, 977, 1172, 1175, 1176, 1178, 1194, 1243, 1739, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1870, 1919, 1960, 1961, 1965, 1967, 2003]


    v_template = v_template - np.mean(v_template)
    y = np.mean(v_template[I,1])
    v_template[:,1] = v_template[:,1] - y
    v_template[I,1] = 0

    

    # Load symmetry data
    symIdx = []
    if animal_type == 'equidae':
        with open(sym_path, 'rb') as f:
            dd = pkl.load(f, encoding='latin1')

        left = v_template[:, 1] < 0
        right = v_template[:, 1] > 0
        center = v_template[:, 1] == 0
        v_template[left[dd['symIdx']]] = np.array([1,-1,1])*v_template[left]

        left_inds = np.where(left)[0]
        right_inds = np.where(right)[0]
        center_inds = np.where(center)[0]
        symIdx = dd['symIdx']
    elif animal_type == 'canidae':
        with open(sym_path, 'rb') as f:
            symmetry_inds_dict = json.load(f)

        left_inds = np.asarray(symmetry_inds_dict['left_inds'])
        right_inds = np.asarray(symmetry_inds_dict['right_inds'])
        center_inds = np.asarray(symmetry_inds_dict['center_inds'])
        v_template[right_inds, :] = np.array([1,-1,1])*v_template[left_inds, :]
        symIdx = center_inds

    try:
        assert(len(left_inds) == len(right_inds))
    except:
        import pdb; pdb.set_trace()

    return v_template, left_inds, right_inds, center_inds, symIdx
