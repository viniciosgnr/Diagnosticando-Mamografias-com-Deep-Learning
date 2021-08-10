# -*- coding: utf-8 -*-




import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import time
import matplotlib.pyplot as plt


import tqdm
from tqdm import tqdm_notebook as tqdm


from sklearn.metrics import confusion_matrix
import CDDSM
 

from early_stopping import EarlyStopping

from lr_scheduler import cyclical_lr



def set_parameter_requires_grad(model,feature_extracting,layer = 0):
     
     if feature_extracting:
          for param in model.parameters():
               param.requires_grad = False
     else:
          ct = 0
          for child in model.children():
               ct += 1
               if ct > layer:
                    for param in child.parameters():
                         param.requires_grad = True
          
          
#Device Selection
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Device:',' GeForce RTX 2070' if torch.cuda.is_available() else 'cpu')

# Hyper parameters

feature_extract = True
num_epochs = 4000
num_classes = 2
batch_size = 64




#Early stopping instance
Early_stopping = EarlyStopping(patience = 30, verbose=True)




# Reading a mammogram
homedir ="C:/Users/CALEXANDRE/Desktop"
#homedir

# CSV preprocessing
train_df = CDDSM.createTrainFrame(homedir)
test_df = CDDSM.createTestFrame(homedir)
mammogram_dir = "C:/Users/CALEXANDRE/Desktop/CuratedDDSM/"
train_file = mammogram_dir+'train.csv'
test_file = mammogram_dir+'test.csv'
train_df.to_csv(train_file)
test_df.to_csv(test_file)

# labells = train_df[['pathology','pathology_class']]
# print(labells)


classes = {'BENIGN': 0, 'MALIGNANT':1}

#Image size
img_resize=H=W=256
     
# Mammography dataset

train_dataset =  CDDSM.MammographyDataset(train_file, homedir, img_resize, "train")
test_dataset = CDDSM.MammographyDataset(test_file,homedir,img_resize,"test")

# Data loader

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


                                          

DataLoaders = {"train": train_loader, "valid": test_loader}

number_of_training_data = train_dataset.__len__()
number_of_testing_data = test_dataset.__len__()

dataset_sizes = {"train" : number_of_training_data, "valid" : number_of_testing_data}

total_step = len(train_loader)


print('Size of training dataset {}'.format(number_of_training_data))
print('Size of testing dataset {}'.format(number_of_testing_data))
print('No. of Epochs: {}\n Batch size: {}\n Image size {}*{}\n Step {}'
        .format(num_epochs,batch_size,H,W,total_step))


model = models.resnet101(pretrained=True)
set_parameter_requires_grad(model, feature_extract)


num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs,256),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(256,num_classes),
                         nn.LogSoftmax(dim=1))
#print(model.children)
model.cuda()



#print('/n', model)
# model = B.getModel(3).to(device)

#defining the weights of our unbalanced dataset
weights = [1, 1.54]
class_weights = torch.FloatTensor(weights).cuda()

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = criterion.cuda()

print("\n *************Training on CBIS-DDSM train data******************\n")



since = time.time()
best_epoch_acc = 0.0

epochs = []
train_loss=[]
valid_loss = []
train_acc = []
valid_acc = []
     
conf_total = np.zeros((2,2))     
# Train the model
for epoch in range(num_epochs):
     conf_total = np.zeros((2,2))
     
     
     
     #from here the model starts the second stage of the trainning with all layers unfreezed
     if epoch == 0 :
          set_parameter_requires_grad(model,False,9)#last layer unfreezed
          lr = 1e-3
          optimizer = torch.optim.SGD(model.parameters(), lr = lr, nesterov=True,
                            momentum=0.9)
          
          scheduler = torch.optim.lr_scheduler.CosineAnnealingWithRestartsLR(optimizer,100,
                                                                   eta_min=8e-4, 
                                                                   last_epoch=-1,T_mult=1)

     
     elif epoch == 6:
          set_parameter_requires_grad(model,False,6)#four top layers unfreezed
          lr = 1e-4
          optimizer = torch.optim.SGD(model.parameters(), lr = lr, nesterov=True,
                            momentum=0.9,weight_decay = 5e-4)
          
          scheduler = torch.optim.lr_scheduler.CosineAnnealingWithRestartsLR(optimizer,100,
                                                                   eta_min=7e-5,
                                                                   last_epoch=-1,T_mult=1)
     elif epoch == 26 :
          set_parameter_requires_grad(model,False,0)#all layers unfreezed
          lr = 7e-5
          
          optimizer = torch.optim.SGD(model.parameters(), lr = lr, nesterov=True,
                            momentum=0.9, weight_decay = 5e-4)
          scheduler = torch.optim.lr_scheduler.CosineAnnealingWithRestartsLR(optimizer,100,
                                                              eta_min=1e-5,
                                                              last_epoch=-1,T_mult=1)
     if Early_stopping.early_stop:
            print("Early stopping")
            break
       
     epochs.append(epoch)
     #each epoch has a train and a valid phase.
     print('Epoch {}/{}'.format(epoch, num_epochs - 1), flush=True)
     print('-' * 10,flush=True)
     
     for phase in ['train','valid' ]:
          
          
          if phase == 'train':
               
               
               #Set the models to training mode
               model.train()
               
          else:
               
               
               #Set the models to evaluation mode
               model.eval()
          
          #Keep a track of all the three loss
          running_loss = 0.0
          
          #Metrics : predictor auc and selector iou
          running_acc = 0

          #tqdm bar
          pbar = tqdm(total= dataset_sizes[phase])
       
          for i, (images, labels) in enumerate(DataLoaders[phase]):
               
               
               try:
                    
                    images = images.float().cuda()
          
                    labels = labels.long().cuda()
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
               except:
                  continue
             
               
               # forward
               # track history if only in train
               with torch.set_grad_enabled(phase == 'train'):
                    
                    #Forward propagation 
                    outputs = model(images)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    
                    #Calculating loss with softmax to obtain cross entropy loss
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                         
                         loss.backward()  # backward propagation   
                         
                         
                         optimizer.step() # updating the gradients 
                         
                         
                         
                    else:
                         
                         scheduler.step() # > Where the magic happens
                         
                         print(labels)
                         print(predicted)
                         conf_matrix = confusion_matrix(labels.cpu().numpy(),
                                                        predicted.cpu().numpy())                              
                         conf_total += conf_matrix
                         print(conf_total)
               # statistics
               
               running_loss += loss.item()* images.size(0)
               running_acc += torch.sum(predicted == labels.data)
               
               print(loss.item())
               pbar.update(images.shape[0])
          pbar.close()
          epoch_loss = running_loss / dataset_sizes[phase]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
          epoch_acc = running_acc.double() / dataset_sizes[phase]
          
          train_loss.append(epoch_loss) if phase == "train" else valid_loss.append(epoch_loss)
          train_acc.append(epoch_acc)  if  phase == "train" else valid_acc.append(epoch_acc)
          
          if phase == "valid":
               Early_stopping(epoch_loss, model)
               
          print('epoch: {},{} Sel_Loss: {:.4f} Acc: {:.4f} '.format(epoch,
                    phase, epoch_loss, epoch_acc))                       
          
          info = 'epoch: {},{} Sel_Loss: {:.4f} Acc: {:.4f} '.format(epoch,
                    phase, epoch_loss, epoch_acc)
          
          f= open("data.txt","a+") #creates the file
          f.write("\n"+info)# writes the info that appear on the terminal within the the txt file
          f.close()


     #Ploting the results 
          
     fig, axs = plt.subplots(2)
     
     axs[0].plot(epochs, train_loss)
     axs[0].plot(epochs, valid_loss)
     
     axs[1].plot(epochs, train_acc)
     axs[1].plot(epochs, valid_acc)
     
     axs[0].legend(("train_loss","valid_loss"), loc="best")
     axs[1].legend(("train_acc","valid_acc"), loc="best")
     
     axs[0].set(xlabel='Epochs', ylabel='Loss',
     title='Learning Progress') 
     axs[1].set(xlabel='Epochs', ylabel='Accuracy')
     

     fig.savefig("Results.png")
     plt.show()    


      
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))


        
print('Training completed finally !!!!!')          
f.close() #close the file  

print("\n ****************Testing on CBIS-DDSM test data*******************\n")

# Test the model
model.load_state_dict(torch.load('checkpoint.pt'))

model.eval()  # eval mode at(bchnorm uses moving mean/variance instead of mini-batch mean/variance)
