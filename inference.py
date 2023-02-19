import torch

from data.test_loader import loader
from utils.model import ModelSA

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

classes = ['TB', 'TE']

# Change image path here
img_path = ''    

# Change weights path, if necessary
model_path = './Weights/best.pth'

# Loading the image
img = loader(img_path)
img = img.to(device)

model = ModelSA()
model = model.to(device)

model.load_state_dict(torch.load(model_path))
model.eval()

pred = model.forward(img)
indx = torch.argmax(pred[0])

print('The prediction is: ', classes[indx])