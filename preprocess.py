#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 22:33:33 2020

@author: alex
"""

import csv
import numpy as np
import time

gendermap = {
    'M': [1,0,0],
    'F': [0,1,0],
    'I': [0,0,1]
}
inputs = []
outputs = []
'''
1. As we need a numerical representation for the first variable, gender,
we can do a three positions one hot encoding, which M will be
[1,0,0], for F, [0,1,0] and for I, [0,0,1].
'''
with open('abalone.csv') as data:
    data = csv.reader(data, delimiter=',')
    for abalone in data:
        if len(abalone) > 0:
            entrada = gendermap[abalone[0]].copy()
            entrada.extend(list(map(lambda x: float(x), abalone[1:-1])))
            inputs.append(entrada)
            outputs.append(int(abalone[-1]))

print(inputs, len(inputs))
