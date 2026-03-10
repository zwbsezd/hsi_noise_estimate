import argparse
#import cv2
import numpy as np
import copy
from os.path import join
from datasetc import fake_rgtinData,realData
from torchlight.utils import instantiate, locate
from torchlight.nn.utils import adjust_learning_rate, get_learning_rate
#import hsir.data.dataloader as loaders
from torch.utils.data import DataLoader
import torch
import models
from hsi_setup import model_names

#torch.optim.lr_scheduler.SequentialLR
#torch.cosine_similarity
# from macnet import MACNet
from models.competing_methods.SST import *
# from T3SC.multilayer import MultilayerModel
# from hsir.model.hsidcnn import HSIDCNN
# from sert import SERT
# from hsdt import hsdt
from models.competing_methods.GRNet import *
from hsdt.arch import HSDT
# from HSIDwRD import U_Net_3D
# from hsir.model import qrnn3d
#F:\SERT-master\models\competing_methods\qrnn\qrnn3d.py
import glob
#from headnoiseconve import headnoise
from hsir.trainer import Trainer
from hsir.scheduler import MultiStepSetLR
def train_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--schedule', type=str, default='hsir.schedule.denoise_default')#denoise_restormer')#denoise_default
    parser.add_argument('--bandwise',default=False)
    #parser.add_argument('--resume',default=True)
    #parser.add_argument('--resume-path', '-rp', type=str, default='F:\HSDT-1.0\checkpoints\hsdt34_1_16_5_1_3_noisenorm2\model_epoch_40_21480.pth')
    #parser.add_argument('--statedict', default='hsdt_m_complex.pth')#'F:\SERT-master\checkpoints\\real_realistic.pth')#t3sc_real_netnew.pth')
    parser.add_argument('--save-root', type=str, default='checkpoints')
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids')
    #parser.add_argument('--name',default='hsdt34_1_16_5_1_3_noisenorm_2')
    cfg = parser.parse_args()
    cfg.gpu_ids =[0]# [int(id) for id in cfg.gpu_ids.split(',')]
    return cfg
class Ident(torch.nn.Module):
    def __init__(self):
        super(Ident, self).__init__()
        self.cv=torch.nn.Conv2d(3,3,3,1,1)
    def forward(self, input):
        return input
class D2_3(torch.nn.Module):
    def __init__(self,model):
        super(D2_3, self).__init__()
        self.net=model
    def forward(self, input):
        input=input[:,None]
        out=self.net(input)
        return out[:,0]

channels=34
def main():

    cfg = train_cfg()
    val_name='rgbfake'
    # trains = open('F:\SERT-master\data_miniimagenet\\train.csv').readlines()
    # dirs = []
    # for t in trains[1:]:
    #     dirs.append('F:\SERT-master\data_miniimagenet\mini-imagenet\images\\'+t[:21])
    # datasetrgb=fake_rgtinData(dirs[:20],channels=3,imsize=32)
    # val_loaderrgb = DataLoader(datasetrgb, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    # dirs=glob.glob('E:\Datas\cutireal\\real34gt'+'*'+'gt.npy')
    # dataset=fake_rgtinData(dirs[0:100],channels=channels,imsize=32)
    # train_loader = DataLoader(dataset,batch_size=16,shuffle=False,drop_last=True,num_workers=0)
    val = np.array([1, 3, 4, 5, 6, 12, 15, 19, 20, 23, 24, 27, 28, 43, 45, 54, 55, 59])
    train = list(np.setdiff1d(np.arange(1, 60), val))
    dataset = realData(train , 'train')
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=0)
    dirs =list(val)

    dataset = realData(dirs,'real')
    val_loaderr = DataLoader(dataset, batch_size=1, shuffle=False,drop_last=False,num_workers=0)
    dataset = realData(dirs*10, 'fake')
    val_loaderf = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    dataset = realData(dirs , 'realwithoutIkb')
    val_loaderrwoikb = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    dataset = realData(dirs*10, 'ori')
    val_loadero = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    #modelnames=['macnet']#train'sert','sst','t3sc',
    #modelnames=['sert','qrnn3d','hsdt','macnet','sst','t3sc','U2d','id']
    modelnames = [ 'remacnet', 'resst', 'ret3sc', 'reU2d','resert', 'reqrnn3d', 'rehsdt',]
    for modelname in modelnames:
        if modelname=='t3sc':
            net = models.__dict__['t3sc_real']()
            # from omegaconf import OmegaConf
            # cfgm = OmegaConf.load('F:\SERT-master\models/competing_methods/T3SC/layers/t3sc_real.yaml')
            # net = MultilayerModel(**cfgm.params)
            state_dict=torch.load('F:\SERT-master\checkpoints/t3sc_real_net.pth')['net']
        elif modelname=='sert':
            net = models.__dict__['sert_real']()
            # net = SERT(inp_channels=34, dim=96, window_sizes=[16, 32, 32], depths=[6, 6, 6], down_rank=8, num_heads=[6, 6, 6],
            #          split_sizes=[1, 2, 4], mlp_ratio=2, memory_blocks=64)#26.7743  0.9111  0.0457
            state_dict = torch.load('F:\SERT-master\checkpoints/real_realistic.pth')['net']
        elif modelname == 'U2d':
            net = models.__dict__['grn_net_real']()
            # net = U_Net_GR(34, 34)
            state_dict = torch.load('F:\SERT-master\checkpoints/GRNET_real_net.pth')['net']
        elif modelname == 'qrnn3d':
            neto = models.__dict__['qrnn3d']()
            # net=qrnn3d.QRNNREDC3D(1, 16, 5, [1, 3], has_ad=True)

            state_dicto = torch.load('F:\SERT-master\checkpoints/qrnn3d_real_net.pth')['net']
            neto.load_state_dict(state_dicto)
            net = D2_3(neto)
            state_dict=copy.deepcopy(net.state_dict())
        elif modelname=='hsdt':
            net = HSDT(1, 16, 5, [1, 3],num_bands=channels)
            state_dict=torch.load('F:\HSDT-1.0\\hsdt_m_complex.pth')
        elif modelname=='macnet':
            net = models.__dict__['macnet']()
            # net = MACNet(in_channels=1,channels=16,num_half_layer=5)
            state_dict=torch.load('F:\SERT-master\checkpoints/macnet_real_net.pth')['net']
        elif modelname=='sst':
            net=models.__dict__['sst_real']()
            # net = SST(inp_channels=34,depths=[6,6,6])
            state_dict=torch.load('F:\SERT-master\checkpoints/sst_real.pth')['state_dict']
        elif modelname=='ret3sc':
            net = models.__dict__['t3sc_real']()
            # from omegaconf import OmegaConf
            # cfgm = OmegaConf.load('F:\SERT-master\models/competing_methods/T3SC/layers/t3sc_real.yaml')
            # net = MultilayerModel(**cfgm.params)
            state_dict=torch.load('F:\SERT-master\checkpoints\\t3sc\model_latest.pth')['net']
        elif modelname=='resert':
            net = models.__dict__['sert_real']()
            # net = SERT(inp_channels=34, dim=96, window_sizes=[16, 32, 32], depths=[6, 6, 6], down_rank=8, num_heads=[6, 6, 6],
            #          split_sizes=[1, 2, 4], mlp_ratio=2, memory_blocks=64)#26.7743  0.9111  0.0457

            state_dict = torch.load('F:\SERT-master\checkpoints\\sert\model_latest.pth')['net']
        elif modelname == 'reU2d':
            net = models.__dict__['grn_net_real']()
            # net = U_Net_GR(34, 34)
            state_dict=torch.load('F:\SERT-master\checkpoints\\U2d\model_latest.pth')['net']
        elif modelname == 'reqrnn3d':
            neto = models.__dict__['qrnn3d']()
            # net=qrnn3d.QRNNREDC3D(1, 16, 5, [1, 3], has_ad=True)

            state_dicto = torch.load('F:\SERT-master\checkpoints\\urbn\qrnn3d_urban.pth')['net']
            neto.load_state_dict(state_dicto)
            net = D2_3(neto)
            state_dict=copy.deepcopy(net.state_dict())
        elif modelname=='rehsdt':
            net = HSDT(1, 16, 5, [1, 3],num_bands=channels)
            state_dict=torch.load('F:\HSDT-1.0\\hsdt_m_complex.pth')
        elif modelname=='remacnet':
            net = models.__dict__['macnet']()
            # net = MACNet(in_channels=1,channels=16,num_half_layer=5)
            state_dict = torch.load('F:\SERT-master\checkpoints\\macnet\model_latest.pth')['net']
        elif modelname=='resst':
            net=models.__dict__['sst_real']()
            # net = SST(inp_channels=34,depths=[6,6,6])
            state_dict=torch.load('F:\SERT-master\checkpoints\\sst\model_latest.pth')['net']
        elif modelname=='id':
            net  = Ident()
            state_dict=net.state_dict()
        else:
            print('model_name erro')
            exit()

        cfg.name=modelname
        schedule = locate(cfg.schedule)
        trainer = Trainer(
            net,
            lr=schedule.base_lr,
            save_dir=join(cfg.save_root, cfg.name),
            gpu_ids=cfg.gpu_ids,
            bandwise=cfg.bandwise,
        )

        trainer.logger.print(cfg)
        #if cfg.resume: trainer.load(cfg.resume_path)
        trainer.net.load_state_dict(state_dict, strict=True)#False)






        metrics = trainer.validate(val_loaderrwoikb, val_name)
        np.save('metrics\\' + modelname + '_realwo_metrics.npy', metrics)
        metrics = trainer.validate(val_loaderf, val_name)
        np.save('metrics\\'+modelname+'_fake_metrics.npy', metrics)
        metrics = trainer.validate(val_loaderr, val_name)
        np.save('metrics\\'+modelname+'_real_metrics.npy', metrics)
        metrics = trainer.validate(val_loadero, val_name)
        np.save('metrics\\'+modelname+'_ori_metrics.npy', metrics)
        #
        #
        # #
        # #if cfg.lr: adjust_learning_rate(trainer.optimizer, cfg.lr)  # override lr
        #
        # lr_scheduler = MultiStepSetLR(trainer.optimizer, schedule.lr_schedule, epoch=trainer.epoch)
        # epoch_per_save = 10
        # best_metrics = {}
        # best_metrics['sam']=1
        # trainer.clip=0.1
        # while trainer.epoch < schedule.max_epochs:
        #     np.random.seed()  # reset seed per epoch, otherwise the noise will be added with a specific pattern
        #     trainer.logger.print('Epoch [{}] Use lr={}'.format(trainer.epoch, get_learning_rate(trainer.optimizer)))
        #     # train
        #     trainer.train(train_loader, warm_up=trainer.epoch == 80)
        #     # save ckpt
        #     if trainer.epoch % 10 == 0:#epoch_per_save
        #         metrics = trainer.validate(val_loaderf, val_name)
        #         if metrics['sam'] < best_metrics['sam']:
        #             best_metrics = metrics
        #             trainer.save_checkpoint('model_best.pth')
        #         trainer.logger.print('best metrics', best_metrics)
        #     if trainer.epoch % epoch_per_save == 0:
        #         trainer.save_checkpoint()
        #     trainer.save_checkpoint('model_latest.pth')
        #     lr_scheduler.step()


if __name__ == '__main__':
    main()

# bo=skimage.segmentation.mark_boundaries(rgt.mean(0),mask)
# plt.imshow((1-bo)[:,:,0]*rgt.mean(0))
#
# out = color.label2rgb(labels, img, kind='avg', bg_label=0)
# import tifffile as tf
# import numpy as np
# import matplotlib.pyplot as plt
# rgt=tf.imread('F:\\real_dataset\gt\\59.tif')/4096
# rin2=rgt+np.sqrt(rgt*9e-7+1e-6)*50*np.random.randn(rgt.shape[0],rgt.shape[1],rgt.shape[2])
# gt9010=np.percentile(rgt.reshape(34,-1),84.13,-1)-np.percentile(rgt.reshape(34,-1),15.87,-1)
# in9010=np.percentile(rin2.reshape(34,-1),84.13,-1)-np.percentile(rin2.reshape(34,-1),15.87,-1)
# plt.plot(gt9010**2)
# plt.plot(in9010**2)
# plt.plot(in9010**2-(rgt.mean(-1).mean(-1)*9e-7+1e-6)*2500*4)
# plt.savefig('temp.jpg')
# plt.close()
