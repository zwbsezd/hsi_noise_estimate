import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.signal as signal

class headnoise(nn.Module):
    def __init__(self,bnet=None,
        dim = 64,laynum=8):
        super(headnoise, self).__init__()
        self.bnet=bnet
        self.conv_first = nn.Sequential(*([nn.Conv3d(2, dim, (7,3,3), 1, (3,1,1),padding_mode='reflect'),nn.ReLU(),]+
                                          [nn.Conv3d(dim, dim, (7, 3, 3), 1, (3, 1, 1),padding_mode='reflect'),nn.ReLU(),]*(laynum-2)+
                                          [nn.Conv3d(dim, 1, (7, 3, 3), 1, (3, 1, 1),padding_mode='reflect'),
                                         ]))

    def forward(self, inp_img):
        x = inp_img[:,0]
        noise=inp_img[:,1]
        noise=torch.sqrt(noise.clip(0))
        n, c, h, w = x.shape
        mins = torch.quantile(x.reshape(n, c, -1), 0.01, -1)
        maxs = torch.quantile(x.reshape(n, c, -1), 0.99, -1)
        subs = 0.5 * (mins + maxs)[:, :, None, None]
        divs = (maxs - mins)[:, :, None, None] + 1e-4
        x = (x - subs) / divs
        noise=noise/divs
        inp= torch.stack([x,noise],1)
        out=self.conv_first(inp)[:,0]
        #out=self.bnet(out)
        x = out + x
        return x * divs + subs

def func(xx,lmin,lmax,scale=3):
    return 1 / (1 + torch.exp(-scale * (xx-lmin)))-1/(1+torch.exp(-scale*(xx-lmax)))
def funk(x):
    return 2*(torch.nn.functional.sigmoid(x*3))
def funb(x):
    return 2*(torch.nn.functional.sigmoid(x)-0.5)
def funk_1(x):
    return -torch.log(2/x-1)/3

class NNLS(nn.Module):
    def __init__(self, N=33,slice=11,pixsize=152, device="cpu",eps=1e-8):
        super().__init__()
        self.device = device
        self.paramk=[torch.nn.Parameter(-torch.ones((1,))).to(self.device) for _ in range(N)]
        self.paramb =[torch.nn.Parameter(torch.zeros((1,))).to(self.device) for _ in range(N)]
        self.paramn = torch.nn.Parameter(torch.ones((1,))*5).to(self.device)
        sigma = slice / 2
        xx = np.arange(-(slice // 2), slice // 2 + 1)
        gauss = 1 / (2 * np.pi * sigma ** 2) * np.exp(- xx ** 2 / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        eyes = torch.eye(N - slice + 1)
        masknbh = torch.zeros(N, N - slice + 1)
        for i in range(slice):
            masknbh[i:i + N - slice + 1] = masknbh[i:i + N - slice + 1] + gauss[i] * eyes
        self.slice=slice
        self.pixsize=pixsize
        self.masknbh=masknbh
        self.eps=eps

    def fitk(self, means,vars,mask,param):
        var_pred = means*param[:,0:1]+param[:,1:2]
        sigma = 4 * var_pred / (self.pixsize)
        out=(vars - var_pred)/sigma
        weight = (mask)#[..., None] * self.masknbh[:, None])
        weight=weight/weight.sum()
        # Pnorm=torch.asarray([0.0013499, 0.02140023, 0.13590512, 0.34134475, 0.34134475,
        #                0.13590512, 0.02140023, 0.0013499])
        # cedge=torch.concatenate([torch.ones(1,)*(-float("inf")),torch.arange(-3,4),torch.ones(1,)*(float("inf"))])
        # cup = out.reshape(-1)[:, None] > cedge[None, :-1]
        # cdown = out.reshape(-1)[:, None] < cedge[None, 1:]
        # Ps_id=(cup&cdown).float()
        # Pnow=Ps_id.sum(0)
        # Pnow=Pnow/Pnow.sum()
        # A=max((Pnow/Pnorm).min(),0.1)
        # P=A*torch.exp(-out.reshape(-1)**2/2)/((Ps_id*Pnorm[None,:]).sum(1))
        # loss=1-(P.clip(0,1)*weight.reshape(-1)).sum()
        ########loss=1-((torch.abs(out)<3).float()*weight).sum()
        loss = 1 - (func(out,-5,5) * weight).sum()
        return loss

    def forward(self, means,vars,mask,param):
        mask=mask.float()
        var_pred = means*param[:,0:1]+param[:,1:2]
        sigma = 4 * var_pred / (self.pixsize)*self.paramn
        out=((vars - var_pred)/sigma).clip(-9,9)
        # weight = (mask)#[..., None] * self.masknbh[:, None])
        # weight=weight/weight.sum()
        Pnorm = torch.asarray([0.3154, 0.4684, 0.6162, 0.7222, 0.7606, 0.7222, 0.6162, 0.4685, 0.3155])
        cedge = torch.concatenate([torch.ones(1, ) * (-10), torch.arange(-6, 7) * 0.5, torch.ones(1, ) * (10)])
        Ps_id = func(out[..., None], cedge[:-6], cedge[6:], 3)
        NN = 5
        Pnow = (Ps_id * mask[..., None])
        iv = torch.sort(means, 1)[1][:, -(means.shape[1] // NN * NN):]
        iu = torch.ones_like(iv) * np.arange(means.shape[0])[:, None]
        aa = Pnow[(iu.cpu().detach().numpy(), iv.cpu().detach().numpy())]
        aa = aa.reshape(means.shape[0], NN, -1, 9)
        Pnow = aa.sum(-2) + 1e-7
        Pnowsum = Pnow.sum(-1)[..., None]
        mask2 = Pnowsum > 50
        Pnow = Pnow / Pnowsum
        gamma=0.001
        ai=(Pnow/Pnorm).clip(2*gamma,1)
        A = -gamma*torch.log(torch.exp(-ai/gamma).sum(-1))
        P = ((A[:, :, None] * Pnorm[None, None,] / Pnow).clip(0, 1) * Pnow)
        # P = ((A * torch.exp(-out**2/2)/np.sqrt(2.0*torch.pi))*0.5) / ((Ps_id * Pnow[None, :]).sum(1))
        loss = 1 - ((P * mask2.float()).sum(-1) / (mask2.float() + 1e-7).sum(-1)).sum() / (mask2.float() + 1e-7).sum()

        return loss

    def fit(self, means,vars, max_iter=500):
        mcopy=means.clone()
        mcopy[mcopy>0.9]=0
        N50=means.shape[1]//10+5
        tk50=torch.topk(mcopy, N50,sorted=False)[1].reshape(-1)
        m50=means[((torch.stack([torch.arange(means.shape[0])]*N50).T.reshape(-1)),tk50)].reshape(means.shape[0],N50)
        v50=vars[((torch.stack([torch.arange(means.shape[0])]*N50).T.reshape(-1)),tk50)].reshape(means.shape[0],N50)

        point_med_means = torch.quantile(m50, 0.25, 1)
        point_med_vars = torch.quantile(v50, 0.25, 1)
        point_25_means = torch.quantile(m50, 0.75, 1)
        point_25_vars = torch.quantile(v50, 0.75, 1)#[self.slice // 2:self.masknbh.shape[1] + self.slice // 2]
        ks = (point_med_vars-self.eps)/point_med_means#((point_med_vars - point_25_vars) / (point_med_means - point_25_means))#*0+self.eps
        #ks = np.median(ks)
        ksmax=(point_med_vars-self.eps)/point_med_means
        ksmin=torch.ones(means.shape[0],)*self.eps
        ks=ks.clip(ksmin,ksmax)
        pks = ks*funk(torch.concatenate(self.paramk))
        bs=point_med_vars - pks*point_med_means
        pbs=bs+point_med_vars*(funb(torch.concatenate(self.paramb)))
        param=torch.concatenate([pks[:,None],pbs[:,None]],1)
        lrk=(ksmax-ksmin)*0.00+1
        lrb = (point_med_vars) *0.00+1
        listop = []
        for i in range(len(lrk)):
            listop.append({'params': [self.paramk[i]], 'lr': lrk[i]})
            listop.append({'params': [self.paramb[i]], 'lr': lrb[i]})
        listop.append({'params': [self.paramn], 'lr': 1})
        optimizer = optim.SGD(listop,)
        lop=[]
        los=[]
        for iteration in range(max_iter):

            zerostd = torch.sqrt(param[:, 1])
            onestd = torch.sqrt(param.sum(1))
            out = self.forward(means, vars,  ((means < (1 - 3 * onestd[:, None])) & (means > 3 * zerostd[:, None])),param)
            lossp = out
            m2 = F.conv1d(param.T.reshape(2, 1, -1), torch.ones_like(param[None, None, :2, 0]), padding=0, stride=1)[:, 0]
            m3 = F.conv1d(param.T.reshape(2, 1, -1), torch.ones_like(param[None, None, :8, 0]), padding=0, stride=1)[:,  0]
            d1 = F.conv1d(param.T.reshape(2, 1, -1), torch.ones_like(param[None, None, :2, 0])*np.array([1,-1]), padding=0, stride=1)[:,  0]
            d2 = F.conv1d(param.T.reshape(2, 1, -1), torch.ones_like(param[None, None, :8, 0])*np.array([1,1,1,1,-1,-1,-1,-1]), padding=0, stride=1)[:,  0]
            lossC1 = (torch.abs(d1) / m2).mean(-1)
            lossC2 = (torch.abs(d2) / m3).mean(-1)
            #gamma=1
            #-1 / (gamma * torch.log((torch.exp(-((1 / lossC1) / gamma))).sum()))
            loss = lossp + ((lossC1[0] + lossC2[0])+(lossC1[1] + lossC2[1]))*0.01
            lop.append(lossp.detach().numpy())
            los.append(loss.detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.paramk+self.paramb, clip_value=0.1)
            optimizer.step()
            ksmax = ((point_med_vars+point_med_vars*(funb(torch.concatenate(self.paramb)))- self.eps) / point_med_means)
            ksmin = torch.ones(means.shape[0], ) * self.eps
            with torch.no_grad():
                ratio=iteration/max_iter
                ratio=max(1-ratio*5,0)
                medk=torch.median(param[:,0])-ks
                for i in range(len(lrk)):
                    self.paramk[i].copy_((self.paramk[i].data*(1-ratio)+medk[i]*ratio).clamp(funk_1(ksmin[i] / ks[i]), funk_1(min(ksmax[i] / ks[i],torch.ones(1,)*1.99))))
                # for i in range(len(lrk)):
                    #self.paramk[i].copy_(self.paramk[i].data.clamp(ksmin[i] - ks[i], ksmax[i] - ks[i]))
                self.paramn.copy_(self.paramn.data.clamp(1,6))
            pks = ks * funk(torch.concatenate(self.paramk))
            bs = point_med_vars - pks * point_med_means
            pbs = bs + point_med_vars*(funb(torch.concatenate(self.paramb)))
            param = torch.concatenate([pks[:, None], pbs[:, None]], 1)
            ttt1 = torch.concatenate([param[1, :], self.paramk[1], self.paramk[1].grad]).detach().numpy()
            # ttt9 = torch.concatenate([param[9, :], self.paramk[9], self.paramk[9].grad]).detach().numpy()
            print(iteration)
            print(ttt1)
            # print(ttt9)

        a=0

        self.param=param

class Ikbcal(nn.Module):
    def __init__(self,nin,ngt,k=50,b=0, device="cpu"):
        super().__init__()
        self.device = device
        self.paramk = torch.nn.Parameter(torch.ones((1,))*np.array([k]).clip(1e-10,)[0])
        self.paramb = torch.nn.Parameter(torch.ones(( 1, )) * np.array([b]).clip(-1,1)[0])
        #torch.tensor([k,b]))#H*W,1,5
        self.paramk = self.paramk.to(self.device)
        self.paramb = self.paramb.to(self.device)
        self.k=k
        self.b=b
        self.nin=nin
        self.ngt=ngt


    def fit(self, rin,rgt, max_iter=1000, lr=None):
        if lr is None:
            lrk=0.1
            lrb=0.001
        else:
            lrk=lr
            lrb=lr
        optimizer = optim.Adam([{'params':[self.paramk], 'lr':lrk},
                                {'params':[self.paramb],'lr':lrb}],
                               )
        #optimizer = optim.Adam([self.paramk]+[self.paramb],lr=1e-7)
        for iteration in range(max_iter):
            iteration += 1
            Ipred = rin * self.paramk + self.paramb
            varis = (self.nin*self.paramk**2+self.ngt)

            out = ((rgt - Ipred) /torch.sqrt(varis.clip(1e-8)).detach()).clip(-20,20)
            times=max(3*(max_iter-1.2*iteration)/(max_iter),0)+1
            s3 = (func(out, -3*times,3*times )).mean()  /0.9889
            sl = (func(out, -1*times, 0) ).mean()
            sr = (func(out, 0, 1*times) ).mean()
            loss = torch.abs(1 - s3) + torch.abs(sl - sr)
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(parameters=[self.paramk]+[self.paramb], max_norm=10, norm_type=2)
            optimizer.step()
            # Force non-negative weights only
            with torch.no_grad():
                self.paramk.copy_(self.paramk.data.clamp(min=1e-10))
        self.s3=s3.cpu().detach().numpy()
# for i in range(1,60):
#     name='real'+str(i).rjust(2,'0')
#     rgt = np.load('tempreal\\regrgt' + name + '.npy')
#     rin = np.load('tempreal\\regrin' + name + '.npy')
#     cc=np.load('temprealkbs\\cc'+name+'.npy')
#     rgtcc=rgt.reshape(rgt.shape[0],-1)[:,cc]
#     rincc=rin.reshape(rgt.shape[0],-1)[:,cc]
#     #mask=(rincc.min(1)>1.5/4096)&(rgtcc.max(1)<0.9)
#     Igt=rgtcc.mean(1).reshape(-1)
#     Iin=rincc.mean(1).reshape(-1)
#     kbsgt=np.load('temprealkbs\\kbs'+name+'_0.npy')
#     kbsin=np.load('temprealkbs\\kbs'+name+'_1.npy')
#     ngt = (rgtcc.mean(1)* kbsgt[:, 0:1] + kbsgt[:, 1:2]).reshape(-1)/(cc.shape[0]-1)
#     nin = (rincc.mean(1) * kbsin[:, 0:1] + kbsin[:, 1:2]).reshape(-1)/(cc.shape[0]-1)
#     ikbs=Ikbcal(torch.tensor(nin),torch.tensor(ngt),50,0.025)
#     ikbs.fit(torch.tensor(Iin),torch.tensor(Igt))
#     np.save('temprealkbs\\Ikbs'+name+'.npy',np.array([ikbs.paramk.cpu().detach().numpy()[0],ikbs.paramb.cpu().detach().numpy()[0],ikbs.s3,Iin.mean()]))
#     a=0

# means = torch.rand(1, 7000) * torch.arange(1, 35)[:, None]
# vars = means * 1e-4 + 4e-5
# # vars = vars * (1 + 12 / 102 * np.random.randn(34, 7000))
# # mask=means*0+1
# means=torch.tensor(np.load('means1728.npy'))
# vars=torch.tensor(np.load('varis1728.npy'))
# mask=(means>(12/4096))&(means<0.95)
# nnls=NNLS(N=33,slice=5,pixsize=152,k=5e-5,b=1e-6, device="cpu")
# nnls.fit(means,vars,mask, max_iter=100, lr=2e-6)
# slice=5
# kbs=torch.nn.functional.pad(nnls.param.T, (slice // 2,slice//2+1), 'replicate').T.detach().numpy()
# a=0

# Ikbs=[]
# for i in range(1,60):
#     name='real'+str(i).rjust(2,'0')
#     Ikbs.append(np.load('temprealkbs\\Ikbs' + name + '.npy'))
# Ikbs=np.stack(Ikbs,0)
#np.intersect1d(np.argsort(-Ikbs[:,2])[:30],np.argsort(-Ikbs[:,3])[:30])+1





