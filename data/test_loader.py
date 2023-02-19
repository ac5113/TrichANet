import torch
import numpy as np
import cv2

def crop(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

classes = ['TB', 'TE']

#Data Loader
def loader(img_path):
    img = cv2.imread(img_path)
    img = crop(img)
    img = cv2.resize(img, (224,224),cv2.INTER_AREA)

    img = np.array(img, np.float32).transpose(2,0,1)/255.0 

    img = torch.tensor(img, dtype = torch.float32)
    img = torch.unsqueeze(img, 0)
    
    return img
