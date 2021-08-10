# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:34:13 2019

@author: Marcos Vinícios 
"""
import cv2
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from gzip import GzipFile







def select_largest_obj(img_bin, lab_val=255, fill_holes=False, 
                       smooth_boundary=False, kernel_size=30):
    n_labels, img_labeled, lab_stats, _ = cv2.connectedComponentsWithStats(
        img_bin, connectivity=8, ltype=cv2.CV_32S)
    largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
    largest_mask[img_labeled == largest_obj_lab] = lab_val
    if fill_holes:
        bkg_locs = np.where(img_labeled == 0)
        bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
        img_floodfill = largest_mask.copy()
        h_, w_ = largest_mask.shape
        mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
        cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed, newVal=lab_val)
        holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
        largest_mask = largest_mask + holes_mask
    if smooth_boundary:
        kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, kernel_)
        
    return largest_mask

def preprocessing(image):
    
    
    
    #thresholding the above result to convert it to black and white
    global_threshold = 18  # from Nagi thesis. <<= para to tune!
    _, mammo_binary = cv2.threshold(image, global_threshold, 
                                maxval=255, type=cv2.THRESH_BINARY)
    #Calling select_largest_obj method which suppresses the artifact number
    mammo_breast_mask = select_largest_obj(mammo_binary, lab_val=255, 
                                       fill_holes=False, 
                                       smooth_boundary=True, kernel_size=10)  # <<= para to tune!
    #Using bitwise and to remove artifact number
    mammo_arti_suppr = cv2.bitwise_and(image, mammo_breast_mask)
    #It enhances the contrast to identify the pectoral bone

    clahe = cv2.createCLAHE(clipLimit = 8, tileGridSize=(8, 8))
    mammo_breast_equ = clahe.apply(mammo_arti_suppr)
    
    #mammo_breast_equ = cv2.equalizeHist(mammo_arti_suppr)
    #Again thresholding the above result to convert to white and black
    pect_high_inten_thres = 200  # <<= para to tune!
    _, pect_binary_thres = cv2.threshold(mammo_breast_equ, pect_high_inten_thres, 
                                     maxval=255, type=cv2.THRESH_BINARY)
    # Markers image for watershed algo.
    pect_marker_img = np.zeros(pect_binary_thres.shape, dtype=np.int32)
    # Sure foreground.
   # pect_mask_init = select_largest_obj(pect_binary_thres, lab_val=255, 
                                    #=False, smooth_boundary=True)
    #kernel_ = np.ones((3, 3), dtype=np.uint8)  # <<= para to tune!
    #n_erosions = 7  # <<= para to tune!
    #pect_mask_eroded = cv2.erode(pect_mask_init, kernel_, iterations=n_erosions)
    #pect_marker_img[pect_mask_eroded > 0] = 255
    # Sure background - breast.
    #n_dilations = 7  # <<= para to tune!
    #pect_mask_dilated = cv2.dilate(pect_mask_init, kernel_, iterations=n_dilations)
    #pect_marker_img[pect_mask_dilated == 0] = 128
    # Sure background - background.
    #pect_marker_img[mammo_breast_mask == 0] = 64
    #Marking and segmenting the image with watershed algorithm'''
    mammo_breast_equ_3c = cv2.cvtColor(mammo_breast_equ, cv2.COLOR_GRAY2BGR)
    cv2.watershed(mammo_breast_equ_3c, pect_marker_img)
    pect_mask_watershed = pect_marker_img.copy()
    #mammo_breast_equ_3c[pect_mask_watershed == -1] = (0, 0, 255)
    #pect_mask_watershed[pect_mask_watershed == -1] = 0
    #breast_only_mask = pect_mask_watershed.astype(np.uint8)
    #breast_only_mask[breast_only_mask != 128] = 0
    #breast_only_mask[breast_only_mask == 128] = 255
    #kn_size = 25  # <<= para to tune!
    #kernel_ = np.ones((kn_size, kn_size), dtype=np.uint8)
    #breast_only_mask_smo = cv2.morphologyEx(breast_only_mask, cv2.MORPH_OPEN, kernel_)
    #mammo_breast_only = cv2.bitwise_and(mammo_breast_equ, breast_only_mask_smo)'''
    
    
    return mammo_breast_equ_3c