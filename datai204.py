import scipy.io
import cv2
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import torch
import torch.nn as nn
import glob
import h5py
def smooth(input):
    dif = input[..., 1:] - input[..., :-1]
    return (dif ** 2).mean()


def residuals(p, y, x):
    k, b = p
    return y + b - k * x


class IkIb(nn.Module):
    def __init__(self,Iki,Ibi ):
        super(IkIb, self).__init__()
        self.Ik = torch.nn.Parameter(torch.zeros((1,)))
        self.Ib = torch.nn.Parameter(torch.zeros((1,)))
        self.Iki=Iki
        self.Ibi=Ibi


    def forward(self):
        Ik = torch.exp(self.Ik) * Iki
        Ib = (F.sigmoid(self.Ib)-0.5)+Ibi
        return Ik, Ib

for imgid in range(51,56):#,1,2,3,4,6,7,8,9,10,51,52,53,54,55]:

    name = str(100 + imgid)
    rin = np.load('F:\SERT-master\models\competing_methods\\rgrin' + str(imgid) + '.npy') / 4096
    rgt = np.load('F:\SERT-master\models\competing_methods\\rgrgt' + str(imgid) + '.npy') /4096
    Iki,Ibi=np.load('F:\SERT-master\models\competing_methods\\kbrgrin' + str(imgid) + '.npy')
    k = np.load('F:\SERT-master\models\in' + str(imgid) + 'k.npy')
    b = np.load('F:\SERT-master\models\in' + str(imgid) + 'b.npy')
    model= IkIb(Iki,Ibi)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, 300)
    losses=[]
    ks=[]
    bs=[]
    for i in range(700):
        Ik,Ib=model()
        c=np.arange(34)
        np.random.shuffle(c)
        hw=np.random.randint(20,100)
        inp=torch.tensor(rin[c[:5],hw:-hw,hw:-hw])
        out=inp*Ik+Ib
        gt=torch.tensor(rgt[c[:5],hw:-hw,hw:-hw])
        mask=(gt >0/4096)&(gt<(1-50/4096))
        loss=((((((out-gt)**2)/(gt*torch.tensor(k)+torch.tensor(b)))*mask).sum(-1).sum(-1)+1e-20)/(mask.sum(-1).sum(-1)+1e-20)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ks.append(Ik.detach().numpy())
        bs.append(Ib.detach().numpy())
        if i<699:
            schedule.step()
        inp = torch.tensor(rin[:,20:-20,20:-20])
        out = inp * Ik + Ib
        gt = torch.tensor(rgt[:,20:-20,20:-20])
        mask = (gt > 0 / 4096) & (gt < (1 - 50 / 4096)) & (inp > (0 / 4096))
        losses.append((((((out-gt)**2)*mask).sum()+1e-10)/(mask.sum()+1e-10)).detach().numpy())
    np.save('F:\SERT-master\models\competing_methods\\kbrgrindl' + str(imgid) + '.npy',np.array([Ik.detach().numpy(),Ib.detach().numpy()]))
    print(str(imgid)+' '+str(np.round(Ik.detach().numpy(),6))+' '+str(np.round(Ib.detach().numpy(),6))+' '+str((((rin*Iki+Ibi)-rgt)**2).mean())+' '+str((((rin*Ik.detach().numpy()+Ib.detach().numpy())[10]-rgt[10])**2).mean()))
    a=0

