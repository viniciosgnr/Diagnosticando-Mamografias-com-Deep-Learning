# -*- coding: utf-8 -*-


from __future__ import print_function, division
import pydicom
import torch
import os
import sys
import pandas as pd
import numpy as np
import pydicom as DCM
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import cv2

from preprocessing import preprocessing



def createTrainFrame(homedir):
    
    train_mass_csv = pd.read_csv(homedir+"/CuratedDDSM/Train/mass_case_description_train_set_1.csv")
    train_calc_csv = pd.read_csv(homedir+"/CuratedDDSM/Train/calc_case_description_train_set_1.csv")
    train_calc_csv = train_calc_csv.rename(columns={'breast density': 'breast_density'})
    train_calc_csv['image file path'] = 'Calc/CBIS-DDSM/'+ train_calc_csv['image file path']
    train_calc_csv['ROI mask file path'] ='CalcROI/CBIS-DDSM/'+ train_calc_csv['ROI mask file path']
    train_calc_csv['cropped image file path'] ='CalcROI/CBIS-DDSM/'+ train_calc_csv['cropped image file path']
    train_mass_csv['image file path'] = 'Mass/CBIS-DDSM/'+ train_mass_csv['image file path']
    train_mass_csv['ROI mask file path'] ='MassROI/CBIS-DDSM/'+ train_mass_csv['ROI mask file path']
    train_mass_csv['cropped image file path'] ='MassROI/CBIS-DDSM/'+ train_mass_csv['cropped image file path']
    common_col = list(set(train_calc_csv.columns) & set(train_mass_csv.columns))
    train = pd.concat([train_mass_csv[common_col],train_calc_csv[common_col]], ignore_index=True,sort='False')
    #train = train_mass_csv
    train['image file path'] = 'CuratedDDSM/Train/'+train['image file path'] 
    train['ROI mask file path'] = 'CuratedDDSM/Train/'+train['ROI mask file path']
    train['cropped image file path'] = 'CuratedDDSM/Train/'+train['cropped image file path']
    train['pathology_class'] = LabelEncoder().fit_transform(train['pathology'])
    #turning into a binary classification task
    train['pathology_class'].replace(to_replace = 1, value =0,inplace =True) 
    train['pathology_class'].replace(to_replace = 2,value = 1,inplace =True)
    #print(train['pathology_class'])
    return train



def createTestFrame(homedir):
    test_mass_csv = pd.read_csv(homedir+"/CuratedDDSM/Test/mass_case_description_test_set.csv")
    test_calc_csv = pd.read_csv(homedir+"/CuratedDDSM/Test/calc_case_description_test_set.csv")
    test_calc_csv = test_calc_csv.rename(columns={'breast density': 'breast_density'})
    test_calc_csv['image file path'] = 'Calc/CBIS-DDSM/'+ test_calc_csv['image file path']
    test_calc_csv['ROI mask file path'] ='CalcROI/CBIS-DDSM/'+ test_calc_csv['ROI mask file path']
    test_calc_csv['cropped image file path'] ='CalcROI/CBIS-DDSM/'+ test_calc_csv['cropped image file path']
    test_mass_csv['image file path'] = 'Mass/CBIS-DDSM/' + test_mass_csv['image file path']
    test_mass_csv['ROI mask file path'] ='MassROI/CBIS-DDSM/' + test_mass_csv['ROI mask file path']
    test_mass_csv['cropped image file path'] ='MassROI/CBIS-DDSM/' + test_mass_csv['cropped image file path']
    common_col = list(set(test_calc_csv.columns) & set(test_mass_csv.columns))
    test = pd.concat([test_mass_csv[common_col], test_calc_csv[common_col]], ignore_index=True,sort='False')
    test['image file path'] = 'CuratedDDSM/Test/'+test['image file path'] 
    test['ROI mask file path'] = 'CuratedDDSM/Test/'+test['ROI mask file path']
    test['cropped image file path'] = 'CuratedDDSM/Test/'+test['cropped image file path']
    test['pathology_class'] = LabelEncoder().fit_transform(test['pathology'])
    #turning into a binary classification task
    test['pathology_class'].replace(to_replace = 1,value = 0,inplace =True)
    test['pathology_class'].replace(to_replace = 2,value = 1,inplace =True)
    #print(test['pathology_class'])
    return test

class MammographyDataset(Dataset):
    """Creating CBIS-DDSM pytorch dataset."""

    def __init__(self, csv_file, root_dir,img_size, option):
        """
        Args:
            csv_file (string): Path to the csv file containing labels.
            root_dir (string): path to CuratedDDSM directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #Image size
        self.img_size = img_size
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.frame = pd.read_csv(csv_file)
        # Image Columns
        self.image_arr = np.asarray(self.frame['cropped image file path'])
        # Labels
        self.label_arr = np.asarray(self.frame['pathology_class'])
        # Calculate Len
        self.data_len = len(self.frame.index)
        # Location of Curated DDSM
        self.root_dir = root_dir
        
        
        #mean = torch.tensor([0.5527, 0.5527, 0.5527]) 
        #std = torch.tensor([0.2110, 0.2110, 0.2110])
        # Normalization step
        #self.normalize = transforms.Normalize(mean=mean, std=std)
     
        
             
        if option == "train":
                
             self.transformations =   transforms.Compose([transforms.Resize((img_size, img_size),
                                                                            interpolation=Image.BICUBIC),
                                                          transforms.RandomHorizontalFlip(p=0.5),
                                                          transforms.RandomVerticalFlip(p=0.5),
                                                          transforms.RandomRotation(degrees=30),
                                                          transforms.CenterCrop(size=224),
                                                          transforms.ColorJitter()])
                                                          
                                                                            
                                                                            
                                                                            
                                                                      
                                                          
                                                          
        else:
             
             self.transformations = transforms.Compose([transforms.Resize(
                       size=256,interpolation=Image.BICUBIC),
                       transforms.CenterCrop(size=224)])  
                                                                      
             
             
             
             
             
             

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # Open image
        #img_as_img = Image.open(single_image_name)
        image_path = os.path.join(self.root_dir,self.image_arr[index])
        #string_image_path = self.root_dir+"/"+self.image_arr[index]
        #print(image_path)
        
        #print (image_path[-4:])
        if image_path[-4:] !='.dcm':
             image_path = image_path[:-2]
        
          
     
        path = image_path[:-5]
        
        aux = path[-5:]
        try:
             if aux == '00000':
                  
                  a = len(str(os.path.getsize(path+'0.dcm')))
                  b = len(str(os.path.getsize(path+'1.dcm')))
             else:
                  a = len(str(os.path.getsize(path[:-2]+'0.dcm')))
                  b = len(str(os.path.getsize(path[:-2]+'1.dcm')))
                  
        except FileNotFoundError :   
             a = 6
             
             
        if  a <= 7  :
             
             if aux == '00000':
                  
                  image_path = path + '0.dcm'
             else:
                  image_path = path[:-2] + '0.dcm'
                  
                  
                 
             
        elif b <= 7:
             
             if aux == '00000':
                  
                  image_path = path +'1.dcm'
             else:
                  image_path = path[:-2] + '1.dcm'
             
        
             
             
             
              
             
        
         
             
                  
                  
       
        image_dcm= DCM.read_file(image_path)
             
             
        
      
             
        image_2d = image_dcm.pixel_array.astype(float)
        image_2d_scaled = (np.maximum(image_2d,0)/ image_2d.max()) * 255.0
        image_2d_scaled = np.uint8(image_2d_scaled)
        
        preprocessed_img = preprocessing(image_2d_scaled)
        
        
        
        #PIL Image 
        img = Image.fromarray(preprocessed_img)
        img_as_img = self.transformations(img)
        #img_as_img.show()
        
        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)
        
        #img_as_tensor_normalized = self.normalize(img_as_tensor)
        
        # Get label(class) of the image based on the cropped pandas column
        image_label = self.label_arr[index]

        return (img_as_tensor, image_label)



class FullTrainingDataset(Dataset):
    def __init__(self, full_ds, offset, length):
        self.full_ds = full_ds
        self.offset = offset
        self.length = length
        # super(FullTrainingDataset, self).init()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.full_ds[i+self.offset]



def trainValSplit(dataset, val_share):
    val_offset = int(len(dataset)*(1-val_share))
    
    return FullTrainingDataset(dataset, 0, val_offset), FullTrainingDataset(dataset, val_offset, len(dataset)-val_offset)

