#!/usr/bin/env python
# coding: utf-8

# In[1]:


script = False
cluster = False

import numpy as np
import pandas as pd
from memory_profiler import memory_usage
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import timm

from sklearn.model_selection import train_test_split
from transformers import ViTFeatureExtractor, ViTForImageClassification

from PIL import Image, ImageOps
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
plt.style.use('dark_background')

if(script): 
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm


# In[2]:


class ImageDataset(Dataset):
    """Dataset of Images, Data and Labels"""

    def __init__(self, images, data, labels, transform=None):
        """
        Create a PyTorch dataset from an array of images
		and an array of labels
        """
        super().__init__()
        self.images = images
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #return python image given the index
        image = self.images[idx]
        new_image = np.empty((3, 45, 45))
        new_image[0], new_image[1], new_image[2] =  self.normalize_image(image)
        new_image = new_image.transpose(1,2,0)
        new_image = Image.fromarray(np.uint8(255*new_image)).convert("RGB")
        label = self.labels[idx]
        data_point = self.data.iloc[idx].to_dict()
        sample = {'image': self.transform(new_image), 'label': label, 'img': image, 'data': data_point}
        return sample
    
    def normalize_image(self, image):
        image_g = (image[0]-np.mean(image[0]))/ np.std(image[0])
        image_r = (image[1]-np.mean(image[1]))/ np.std(image[1])
        image_i = (image[2]-np.mean(image[2]))/ np.std(image[2])

        image_g = (image_g-np.min(image_g))/ (np.max(image_g) - np.min(image_g))
        image_r = (image_r-np.min(image_r))/ (np.max(image_r) - np.min(image_r))
        image_i = (image_i-np.min(image_i))/ (np.max(image_i) - np.min(image_i))
        return(image_i, image_r, image_g)
    
    def plot_image(self, idx):
        image = images[idx]
        new_image = np.empty((3, 45, 45))
        new_image[0], new_image[1], new_image[2] =  self.normalize_image(image)
        new_image = new_image.transpose(1,2,0)
        new_image = Image.fromarray(np.uint8(255*new_image)).convert("RGB")
        #new_image = Image.fromarray(np.uint16(255*new_image)).convert("RGB")
        
        plt.figure(figsize=(12,4)) 
        
        plt.subplot(1,2,1)
        plt.imshow(np.asarray(new_image))
        plt.axis('off')
        
        plt.subplot(1,2,2)
        rgb = make_lupton_rgb(image[2], image[1], image[0], Q=11., stretch=40.)
        plt.imshow(rgb, aspect='equal')
        plt.axis('off')
        
        plt.show()


# In[3]:


def make_train_test_datasets(images, data, labels, test_size=0.2, transform=None):
    """
	Make training and testing datasets
	
	Args:
	    images: 3D array of all images
        labels: 1D array of the labels for each image
        test_size: the fraction of the images to use as the test dataset
		transform: the PyTorch transformation to apply to the data
		
	Returns
	    train_dataset: An instance of the ImageDataset Class for training
		test_dataset: An instance of the ImageDataset Class for testing
	"""

    # Shuffle and split data
    y = labels
    train_images, test_images, train_data, test_data, train_labels, test_labels = train_test_split(
        images, data, labels, test_size=test_size, random_state=6, stratify=y)
    
    # Create a PyTorch Dataset
    return (ImageDataset(train_images, train_data, train_labels, transform=transform),
            ImageDataset(test_images, test_data, test_labels, transform=transform))


# In[4]:


class ViTBase16(nn.Module):
    
    def __init__(self, n_classes, pretrained=False):
        
        super(ViTBase16, self).__init__()
        
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
        if (pretrained):
            path = 'jx_vit_base_p16_224-80ecf9dd.pth'
            self.model.load_state_dict(torch.load(path))

        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x


# In[5]:


def search_tile(filename, prob_lim):
    file_path = path + filename
    print(filename)
    
    hdu_list = fits.open(file_path)
    search_ids = pd.DataFrame(hdu_list[1].data)
    print(len(search_ids))
    search_labels = 2*np.ones(len(search_ids), dtype = np.int64)
   
    int_arr = hdu_list[2].data    # change 2->3 for PSFs 
    img_min = hdu_list[4].data    # change 4->6 for PSFs
    img_scale = hdu_list[5].data  # change 5->7 for PSFs
    search_images = int_arr / 65535 * img_scale[:,:,np.newaxis,np.newaxis] + img_min[:,:,np.newaxis,np.newaxis]

    search_dataset = ImageDataset(search_images, search_ids, search_labels, transform=transform)
    search_loader = torch.utils.data.DataLoader(dataset=search_dataset, batch_size=1, num_workers=num_workers, shuffle=True)
    
    positives = np.zeros((1,4,45,45))
    data_df = pd.DataFrame()
    
    for i_batch, sample in enumerate(tqdm(search_loader)):
        #if(i_batch==5): break
        sample_image, sample_label, sample_img, sample_data = sample['image'], sample['label'] , sample['img'], sample['data']
        
        output = model(sample_image)
        predicted = output.argmax(dim=1).item()
    
        prob = nn.Softmax(dim=1)(output)
        prob = prob[:,0].detach().numpy()[0]
    
        predicted = 0 if prob >= prob_lim else 1
    
        if(predicted == 0):
            new_df = pd.DataFrame.from_dict(sample_data)
            new_df['Prob'] = prob
            data_df = data_df.append(new_df, ignore_index=True)
            positives = np.append(positives, [np.array(sample_img[0])], axis = 0)
    
    positives = np.delete(positives, 0, axis = 0)
    
    if(len(positives) > 0):
        print('Found {} single lenses in this tile.'.format(len(data_df)))
        primary = fits.PrimaryHDU()
        image = fits.ImageHDU(positives, name="IMAGE")
        data_df = data_df.astype({'COADD_OBJECT_ID': int, 'Prob': float})
        data_table = Table.from_pandas(data_df)
        table = fits.BinTableHDU(data = data_table)
        hdu_list = fits.HDUList([primary, image, table])
        hdu_list.writeto('Y6_detections/' + filename[:-5] + '.fits', overwrite=True)


# In[6]:


if(not(cluster)): 
    path = '/Users/jimenagonzalez/research/DSPL/Searching-double-lenses/vit_cnn/Y6_catalog_files/'
    num_workers = 0
else:
    path = '/data/des81.b/data/stronglens/Y6_CUTOUT_IMAGES/'
    num_workers = 16

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

model = timm.create_model("vit_base_patch16_224", pretrained=True)
path_model = 'jx_vit_base_p16_224-80ecf9dd.pth'
model.load_state_dict(torch.load(path_model))

model = ViTBase16(n_classes=2, pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

name = 'exp_39/model.pt'#'model.pt'#'other.pt' 
model = torch.load(name)


# In[7]:


filename = 'DES2359-6331.fits'
if(cluster):
    filename = sys.argv[1]
prob_lim = 0.9
mem_usage = memory_usage((search_tile, (filename, prob_lim )))
print('Maximum memory usage: %s' % max(mem_usage))

