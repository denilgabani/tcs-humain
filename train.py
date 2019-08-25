# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 23:43:26 2019

@author: DG
"""

from emotions import train_emotions
from age import train_age
from ethnicity import train_ethnicity
from gender import train_gender

#calling training functions from all training files
em = train_emotions()
a = train_age()
eth = train_ethnicity()
g = train_gender()






