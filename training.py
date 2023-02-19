import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from data.loader import TrichDataset
from utils.model import ModelSA
from utils.loss import FocalLoss

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class MyFrame():
    def __init__(self, net, learning_rate, device, evalmode=False):
      self.net = net.to(device)
      self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=learning_rate, weight_decay=0.0001)
      self.loss = FocalLoss().to(device)
      self.lr = learning_rate
        
    def set_input(self, img_batch, label_batch):
      self.img = img_batch
      self.mask = label_batch.float()
        
    def optimize(self):
      self.optimizer.zero_grad()
      pred = self.net.forward(self.img).float()
      loss = self.loss(pred, self.mask)
      loss.backward()
      self.optimizer.step()
      return loss, pred
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.lr / new_lr
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=new_lr, weight_decay=0.0001)

        print ('update learning rate: %f -> %f' % (self.lr, new_lr))
        print ('update learning rate: %f -> %f' % (self.lr, new_lr))
        self.lr = new_lr  

def trainer(epoch, epochs, train_loader, solver, smart=False):
    keep_training = True
    no_optim = 0
    train_epoch_best_loss = INITAL_EPOCH_LOSS
    prev_loss = 1
    print('Epoch {}/{}'.format(epoch, epochs))
    predlist = []
    masklist = []
    train_epoch_loss = 0
    length = len(train_loader)
    iterator = tqdm(enumerate(train_loader), total=length, leave=False, desc=f'Epoch {epoch}/{epochs}')
    for index, (img, mask) in iterator :
        img = img.to(device)
        mask = mask.to(device)
        solver.set_input(img, mask)
        train_loss, p = solver.optimize()

        p = torch.argmax(p, 1)
        mask = torch.argmax(mask, 1)

        predlist = predlist + p.detach().cpu().numpy().tolist()
        masklist = masklist + mask.detach().cpu().numpy().tolist()
        
        train_epoch_loss += train_loss
    
    length = len(train_dataset)/batch_size
    
    train_epoch_loss = train_epoch_loss/length

    cm = confusion_matrix(masklist, predlist)

    print('Confusion matrix: ')
    print(cm)

    print('Classification Report: ')
    print(classification_report(masklist, predlist, target_names=classes))

    print('train_loss:', train_epoch_loss)
    print('Learning rate: ', solver.lr)

    print('---------------------------------------------')
    return train_epoch_loss, keep_training

def tester(model, test_loader, test_best_loss, solver, num, logfile=None, smart=True):
    test_loss = 0
    test_pred = []
    test_mask = []
    Loss_console = nn.CrossEntropyLoss()
    with torch.no_grad() : 
        length = len(test_loader)
        iterator = tqdm(enumerate(test_loader), total=length, leave=False, desc='Testing...')
        for index, (img, mask) in iterator :
            img = img.to(device)
            mask = mask.to(device).float()
            p1 = model.forward(img).float()
            
            loss = Loss_console.forward(p1, mask)

            p1 = torch.argmax(p1, 1)
            mask = torch.argmax(mask, 1)

            test_pred = test_pred + p1.detach().cpu().numpy().tolist()
            test_mask = test_mask + mask.detach().cpu().numpy().tolist()
            
            test_loss += loss

        test_loss = test_loss/(len(test_dataset))

        test_cm = confusion_matrix(test_mask, test_pred)

        print('Test Confusion matrix: ')
        print(test_cm)

        print('Test Classification Report: ')
        print(classification_report(test_mask, test_pred, target_names=classes))

        print('Test loss : ', test_loss)

        if smart==True:
            if test_loss >= test_best_loss: 
                num+=1
                print('Not Saving model: Best Test Loss = ', test_best_loss, ' Current test loss = ', test_loss)
            else:
                num=0
                print('Saving model...')
                solver.save(model_path)
                test_best_loss = test_loss

            if num>=3: 
                num = 0
                print('Loading model...')
                solver.load(model_path)
                solver.update_lr(2, True)
        
        print('---------------------------------------------')
        return test_loss, test_best_loss, num

def train(init=True):
    solver = MyFrame(model, learning_rate, device)

    tr_Loss = []
    EP = []
    te_Loss = []
    
    start_ep = 0
    if len(EP)!=0: start_ep = EP[-1]
    tbl = 100000000000
    num=0

    for epoch in range(start_ep+1, epochs + 1):
        
        logfile = None

        l, k = trainer(epoch, epochs, train_loader, solver, logfile)
        tr_Loss += [l]
        EP += [epoch]

        tr_comp = [tr_Loss, EP]

        lt, tbln, num = tester(solver.net, test_loader, tbl, solver, num, logfile)
        if tbln<=tbl: tbl = tbln
        te_Loss += [lt]

        te_comp = [te_Loss]

        if k: continue
        else: break

if __name__ == '__main__':
    # Dataset path, model path and hyperparameters
    root_path = ''
    model_path = ''
    batch_size = 8

    learning_rate = 0.0002
    epochs = 20

    INITAL_EPOCH_LOSS = 10
    NUM_EARLY_STOP = 20
    NUM_UPDATE_LR = 3

    train_dataset = TrichDataset(root_path, 'Train')
    test_dataset = TrichDataset(root_path, 'Test')

    print(len(train_dataset))
    print(len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    model = ModelSA()
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Parameters: ", pytorch_total_params)

    train()