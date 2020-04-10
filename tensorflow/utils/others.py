
"""
load file

"""

import numpy as np
import h5py
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import random
random.seed(15213) 

def load_files(folder_name, ext):
    """                                                                                       
    used to load the hdf5 file                                                                
    """
    list_in, list_out = [], []
    logging.info("Starting to loadding files ... ")
    for element in os.listdir(folder_name):
        logging.info("File:"+ element)
        hdata = h5py.File(os.path.join(folder_name, element), "r")
        ins, outs = hdata["input"][()], hdata["output"][()]
        logging.info("applying old normalization")
        list_in.append(ins)
        list_out.append(outs)
        hdata.close()
    return np.concatenate(list_in, axis=0), np.concatenate(list_out, axis=0)
