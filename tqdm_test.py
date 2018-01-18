#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:35:12 2018

@author: jmj136
"""

from tqdm import trange
from random import random, randint
from time import sleep

t = trange(100)
for i in t:
    # Description will be displayed on the left
    t.set_description('GEN %i' % i)
    # Postfix will be displayed on the right, and will format automatically
    # based on argument's datatype
    t.set_postfix(loss=random(), gen=randint(1,999), str='h', lst=[1, 2])
    sleep(0.1)