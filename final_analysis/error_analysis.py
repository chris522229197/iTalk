#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import numpy as np
import normalization_helpers as nh

with open('final_analysis/svm_dev.p', 'rb') as f:
    svm_dev = pickle.load(f)

# Plug in ground truth for component 1
comp_1 = svm_dev[svm_dev['predicted_class'] == 'NONE']
comp_2_3 = svm_dev[svm_dev['predicted_class'] != 'NONE']
comp1_truth = len(comp_1) + len(comp_2_3[comp_2_3['predicted_after'] == comp_2_3['after']])
comp1_truth_acc = comp1_truth / len(svm_dev)

# Plug in ground truth for component 1 and 2
comp_2_3['comp2_predicted_after'] = comp_2_3.apply(lambda r: nh.normalize_token(r['before'], r['class']), 
        axis=1)
comp2_truth = len(comp_2_3[comp_2_3['after'] == comp_2_3['comp2_predicted_after']])

comp2_truth_acc = (len(svm_dev) - len(comp_2_3) + comp2_truth) / len(svm_dev)

