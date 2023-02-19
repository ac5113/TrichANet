import torch
import argparse

from data.test_loader import loader
from utils.model import ModelSA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", help="path of the image to be checked")
    parser.add_argument("--model_path", help="path of the pth file", default='./Weights/best.pth')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    classes = ['TB', 'TE']

    # Change image path here
    img_path = args.img_path   

    # Change weights path, if necessary
    model_path = args.model_path

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