#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Testing Branching """

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import nngt

import dense as ds
from dense.units import *

def functions():
    '''
    
    '''
    ds.get_kernel_status('environment_required', False)
    m  = ds.generate_model('constant', 'memory_based', 'run-and-rumble')
    print(m)
    dp = ds.get_default_parameters(m)
    e  = ds.get_environment()
    ks = ds.get_kernel_status()
    ms = ds.get_models()
    gn = ds.create_neurons(1, m)
    n  = ds.get_neurons()
    ns = ds.get_object_status(n)
    si = ds.get_simulation_id()
    ds.simulate(20*hour)
    ni = ds.get_neurons()
    st = ds.NeuronStrucuture(n)
    ds.reset_kernel()
    return 1

def test_functions():
    assert functions() == 1

if __name__ == '__main__':
    functions()