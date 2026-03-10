from torch.utils.data import DataLoader, Dataset
from PIL import Image
import scipy
import scipy.interpolate
import noisemask
#from MLclf import MLclf
ns=noisemask.AddNoiseMixed([noisemask.AddNoiseNoniid((6e-5,1e-3), (1e-6,6e-4)),
                            noisemask.AddNoiseImpulse(amounts=0.001, dots=0.1, s_vs_p=0.5),
                            noisemask.AddNoiseStripe(amount=0.05),
                            noisemask.fliprot(True)],
                           [1.,0.8,0.8,0.8])
Noisemix=noisemask.AddNoiseMixed([noisemask.AddNoise((5e-4), (3e-4)),#AddNoise(4e-3, 3e-3),
                            noisemask.AddNoiseImpulse(amounts=0.001, dots=0.1, s_vs_p=0.5),
                            noisemask.AddNoiseStripe(amount=0.05),],
                                 [1.,1.,1]
                                 )

#MLclf.miniimagenet_download(Download=True)
# from scipy.ndimage import filters
# import glob
import numpy as np
class fake_rgtinData(Dataset):
    def __init__(self, dirs,channels=34,ns=ns,imsize=64,mode='train',noise=True,line='temp34.npy'):
        self.dirs=dirs
        self.channels=channels
        self.mode=mode
        self.imsize=imsize
        self.noise=noise
        self.line=np.load(line)
        self.transform=ns
    def __len__(self):  # 返回整个数据集的大小
        return len(self.dirs)
    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        if '.npy' in self.dirs[index]:
            if self.mode=='train':
                rgt = np.load(self.dirs[index]).astype(np.float32)
            else:
                rgt = np.load(self.dirs[index]).astype(np.float32)[:, 100:612, 4:516]

        else:
            rgt = Image.open(self.dirs[index])
            rgt= np.array(rgt) .transpose(2,0,1).astype(np.float32)/255*0.95+0.05
            np.random.shuffle(rgt)
        c, h, w = rgt.shape
        if (h < 1.5*self.imsize) or (w < 1.5*self.imsize):
            rep=int(np.ceil(self.imsize*1.5/2/min(h,w)))
            rgt=np.concatenate([rgt,rgt[:,::-1]]*rep,1)
            rgt=np.concatenate([rgt,rgt[:,:,::-1]]*rep,2)

        c = self.channels
        crop =self.imsize
        C, H, W = rgt.shape


        if self.mode=='train':
            s1 = np.random.randint(0, H -  crop + 1)
            s2 = np.random.randint(0, W -  crop + 1)
            rgt = rgt[:, s1:s1 + int(1 * crop), s2:s2 + int(1 * crop)]

            if C >=c:
                cc =np.arange(C)
                np.random.shuffle(cc)
                cc=cc[:c]
                cc.sort()
                ii=np.random.randint(c)
                cc[list(cc[ii:]) + list(cc[:ii])]
                rgt=rgt[cc]
                #np.random.permutation(range(W))
            else:
                ti=np.random.randint(0,self.line.shape[1])
                c0=self.line[:,ti]
                cc = np.arange(C)
                ii = np.random.randint(0, C)
                cc = cc[list(cc[ii:]) + list(cc[:ii])]
                rgt = rgt[cc]
                rr=np.random.randint(3)
                if rr==0:
                    interp = scipy.interpolate.interp1d(np.arange(C)/(C-1), rgt,'quadratic',axis=0)
                elif rr==1:
                    interp = scipy.interpolate.interp1d(np.arange(C) / (C - 1), rgt, 'linear', axis=0)
                else:
                    interp = scipy.interpolate.interp1d(np.arange(C) / (C - 1), rgt, 'nearest', axis=0)
                rgt=interp(np.arange(c) / (c-1))
                rgt=(rgt)/(rgt.mean(-1).mean(-1)[:,None,None]/c0[:,None,None])

            kk = 50
            k = (6.5e-5) * kk
            b = (1.2e-6) * kk ** 2
            # kk=np.random.rand()*10+45
            # k = (6e-5 + 1e-5 * np.random.rand()) * kk
            # b = (1e-6 + 0.4e-6 * np.random.rand()) * kk ** 2
        else:
            n1 = (H // crop )
            n2 = (W // crop )
            s1=(H-n1*crop)//2
            s2=(W-n2*crop)//2

            #rgt = rgt[:c, s1:s1 + n1*crop, s2:s2 + n2*crop].reshape(c,n1,crop,n2,crop).transpose(1,3,0,2,4).reshape(-1,c,crop,crop).reshape(-1,crop,crop)
            #rgt = (rgt) / (rgt.max()) * 0.9
            kk = 50
            k = (6.5e-5 ) * kk
            b = (1.2e-6) * kk ** 2
        rin=self.transform(rgt)
        ############rin=rgt+np.random.randn(rgt.shape[0],rgt.shape[1],rgt.shape[2])*np.sqrt((rgt*k+b).clip(0))
        #if self.mode == 'train':
            #rin=self.transform(rin)
        data={}
        # if self.noise:
        #     rin=np.stack([rin,rin*k+b],0)
        #     data['input'] = np.ascontiguousarray(rin[None].astype(np.float32))
        #     data['target'] = np.ascontiguousarray(rgt[None].astype(np.float32))
        # else:
        data['input']=np.ascontiguousarray(rin.astype(np.float32))
        data['target']=np.ascontiguousarray(rgt.astype(np.float32))

        return data

class realData(Dataset):
    def __init__(self, dirs,mode='real'):
        self.dirs=dirs
        self.mode=mode
    def __len__(self):  # 返回整个数据集的大小
        return len(self.dirs)
    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        name=str(self.dirs[index]).rjust(2,'0')


        if self.mode == 'train':
            s1 = np.random.randint(0, 696 - 256 + 1)
            s2 = np.random.randint(0, 512 - 256 + 1)
            rgt = np.load('F:\HSDT-1.0\\data/real/regrgtreal' + name + '.npy').astype(np.float32)[:, s1:s1 + 256, s2:s2 + 256]
            Ikb = np.array([50, 0])
            kb = np.array([[6.5e-5, 1.2e-6]])
            rin = (rgt - Ikb[1]) / Ikb[0] + np.random.randn(rgt.shape[0], rgt.shape[1], rgt.shape[2]) * np.sqrt(
                (((rgt - Ikb[1]) / Ikb[0]) * kb[:, 0, None, None] + kb[:, 1, None, None]).clip(0))
        else:
            rgt = np.load('F:\HSDT-1.0\\data/real/regrgtreal' + name + '.npy').astype(np.float32)[:, 100:356, 4:260]
            if self.mode == 'realwithoutIkb':
                rin = np.load('F:\HSDT-1.0\\data/real/regrinreal' + name + '.npy').astype(np.float32)[:, 100:356, 4:260]
                Ikb=np.array([50,0])


            elif self.mode=='real':
                rin=np.load('F:\HSDT-1.0\\data/real/regrinreal'+name+'.npy').astype(np.float32)[:, 100:356, 4:260]
                Ikb=np.load('F:\HSDT-1.0\\temprealkbs\Ikbsreal'+name+'.npy')
                #kb=np.load('F:\HSDT-1.0\\temprealkbs\\kbsreal01_1.npy')

            elif self.mode=='ori':
                Ikb = np.array([1, 0])
                kb = np.array([[6.5e-5, 1.2e-6]])
                rin = (rgt - Ikb[1]) / Ikb[0] + np.random.randn(rgt.shape[0], rgt.shape[1], rgt.shape[2]) * np.sqrt(
                    (((rgt - Ikb[1]) / Ikb[0]) * kb[:, 0, None, None] + kb[:, 1, None, None]).clip(0))
            elif self.mode=='fake':
                Ikb=np.array([50,0])
                kb=np.array([[6.5e-5,1.2e-6]])
                rin=(rgt-Ikb[1])/Ikb[0]+np.random.randn(rgt.shape[0],rgt.shape[1],rgt.shape[2])*np.sqrt((((rgt-Ikb[1])/Ikb[0])*kb[:,0,None,None]+kb[:,1,None,None]).clip(0))
            else:
                print('mode error')
                exit()
        data={}
        data['input'] = (np.ascontiguousarray(rin.astype(np.float32)) * Ikb[0] + Ikb[1]).clip(0)
        data['target'] = np.ascontiguousarray(rgt.astype(np.float32))

        # data['input'] = np.ascontiguousarray(np.stack([rin,rin*kb[:,0,None,None]+kb[:,1,None,None]],0).astype(np.float32))*Ikb[0]+Ikb[1]
        # data['target'] = np.ascontiguousarray(rgt[None].astype(np.float32))
        return data