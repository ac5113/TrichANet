import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import os

classes = ['TB', 'TE']

def crop(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

#Data Loader
def loader(img_path, phase):
    img = cv2.imread(img_path)
    img = crop(img)
    img = cv2.resize(img, (224,224),cv2.INTER_AREA)

    img = np.array(img, np.float32).transpose(2,0,1)/255.0 
    
    return img

def read_dataset(root_path, mode):
    images = []
    labels = []

    image_root = os.path.join(root_path, mode)

    for label_name in sorted(os.listdir(image_root)):
      label_path = os.path.join(image_root, label_name)
      label = [x==label_name for x in classes]
      label = list(map(int,label))
      for image_name in sorted(os.listdir(label_path)):
        image_path = os.path.join(label_path, image_name)
        images.append(image_path)
        labels.append(label)
     
    return images, labels

class TrichDataset(Dataset):

    def __init__(self, root_path, phase):
        self.root = root_path
        self.phase = phase
        self.images, self.labels = read_dataset(self.root, self.phase)

    def __getitem__(self, index):
        
        img = loader(self.images[index], self.phase)
        img = torch.tensor(img, dtype = torch.float32)
        label = self.labels[index]
        label = torch.tensor(label)
        return img, label

    def __len__(self):
        return len(self.images)