from tensorflow.keras.models         import model_from_json,  model_from_json
from tensorflow.compat.v1.keras import backend as K
import math, os, gc, sys, joblib, pandas as pd, csv, numpy as np, tensorflow as tf


def keras_init():
    print( f"num gpus available {len(tf.config.experimental.list_physical_devices('gpu'))}" )
    # #################################################################### HARDWARE CONFIG
    GPU = False
    CPU = True
    NUM_CORES = 4
    if GPU:
        num_GPU = 1
        num_CPU = 2
    if CPU:
        num_CPU = 2
        num_GPU = 0
    _config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads = NUM_CORES,
                                       inter_op_parallelism_threads = NUM_CORES, 
                                       allow_soft_placement         = True,
                                       device_count                 = {'CPU' : num_CPU, 'GPU' : num_GPU})
    K.set_session(tf.compat.v1.Session(config = _config))
    physical_devices = tf.config.list_physical_devices('GPU')
    if(len(physical_devices) > 0):
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    gc.collect()
    return None

keras_init()

gc.collect()