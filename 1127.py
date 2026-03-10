import scipy.io
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import scipy.signal
from sklearn import linear_model, datasets
import sklearn
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import lsq_linear
from multiprocessing import Pool,Manager
import time
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import glob
import h5py

import pandas as pd
from scipy.stats import norm
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def fun0(x):
    x=np.clip(x,1e-20,0.9535999228020248-1e-20)
    a0,e0,e1,e2,a3,e3,a4,e4=[ 0.18793214, -0.70354188,  2.07627264,  2.49554418,  0.80035301  ,2.45190611, -1.80092321 , 2.47523914]
    y_x=((a0*x**e0*(0.9535999228020248-x)**e1)+0.08033)*(x**e2+a3*x**e3+a4*x**e4)/(0.9535999228020248**e2+a3*0.9535999228020248**e3+a4*0.9535999228020248**e4)
    return y_x
def weightequal(x):
    x = x / 0.9893590245154923
    a0, e0, e1 = [0.79753666, 0.52331227, 6.16582711]
    y_x = (a0 * (x) ** e0 + (1 - a0) * x ** e1) * 0.9534883879591118
    return y_x
def weightref( x0, x1):
    x1 = np.clip(x1, 1e-10, 0.9535999228020248 - 1e-10)
    zerob = fun0(x1)
    x = np.clip(x0, zerob , 0.08033 )
    x=(x-zerob)/(0.08033-zerob)*0.08033
    t1,t2,t3,t4,t5,t6,t7 ,t8= [29, 0.6, 0.65, 0.96536857, 1.76963617, 0.63343317, -0.47162632, 0.9054986]
    a0=np.clip(-t1+x1*t1/t2,0,-t1)
    e0=t3
    e1=t4*x1**t5+t6
    e2=t7*x1+t8
    y_x = ((a0 * (x) ** e0 * (0.08033 - x) ** e1) + 0.9535999228020248) * (
            (x) ** e2 ) / ((0.08033) ** e2) #*b+(1-b)*(x)*  0.9535999228020248/ 0.08033
    return y_x

def epps_pulley_test(x, alpha=0.05):
    x = np.array(x)
    x_bar = np.mean(x)
    m2 = np.var(x)
    n = len(x)
    if n < 10:
        return False
    Ai=np.stack([x]*n,0)
    Ak=np.stack([x]*n,-1)
    A=np.exp( -(Ai-Ak)**2/ (2 * m2)).sum()/2-n/2

    A = 2 / n * A
    B = -np.sqrt(2) * np.sum(np.exp(-(x - x.mean()) ** 2 / (4 * m2)))
    Tn = 1 + n / np.sqrt(3) + A + B


    # 博客参考文献三的第4小节中的动态检验步骤的代码实现
    Tn_star = (Tn - 0.365 / n + 1.34 / (n ** 2)) * (1 + 1.3 / n)
    gamma = 3.55295
    delta = 1.23062
    lam = 2.26664
    xi = -0.020682
    # 以自然数 e 为底
    Zn = gamma + delta * np.log((Tn_star - xi) / (xi + lam - Tn_star))
    if np.median(x)>0.2:
        critical_value = 1.28#norm.ppf(0.9)
    elif np.median(x)>0.1:
        critical_value = 1.65
    else:
        critical_value = 2.3
    test_statistic = round(Zn, 2)
    return test_statistic < critical_value
def Fun(p,x):                        # 定义拟合函数形式
    a1,a2 = p
    return a1*x+a2
def error (p,x,y):                   # 拟合残差
    return ((Fun(p,x)-y))

def round_sf_np(x,significant_figure=0):
    r=np.ceil(np.log(x)/np.log(10))
    f=significant_figure
    return np.round(x*(10**(f-r)),0)*(10**(r-f))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
def Fun(p,x):                        # 定义拟合函数形式
    a1,a2 = p
    return a1*x+a2
def error (p,x,y):                   # 拟合残差
    return (Fun(p,x)-y)
# ss=[]
# dcsum=0
# dncsum=0
# for i in np.arange(1000):
#     na=0
#     nnc =1/np.sqrt(1+na**2)
#     nna=nnc*na
#     nnb=nnc#np.random.rand()**3*5
#     A=np.random.randn(100000,)*nna
#     B=A+np.random.randn(100000,)*nnb
#     C=A+np.random.randn(100000,)*nnc
#     min30C=np.percentile(C,30)
#     max70C=np.percentile(C,70)
#     min30B = np.percentile(B, 30)
#     max70B = np.percentile(B, 70)
#     B=(B-min30B)/(max70B-min30B)
#     C=(C-min30C)/(max70C-min30C)
#     maskdif=(C>=0)&(C<=1)
#     tt=leastsq(error, (1, 0), (B[maskdif], C[maskdif]))[0]
#     tempD=C-(tt[0]*B+tt[1])
#     db=B[maskdif].std(ddof = 1)**2
#     dc=C[maskdif].std(ddof = 1)**2
#     de=tempD[maskdif].std()**2
#     dcsum=dcsum+dc
#     dncsum=dncsum+nnc/(max70C-min30C)
#     #plt.plot(na,tt[maskdif].std()**2-db,'.')
#     ss.append(np.array([de,nnb/(max70B-min30B),nnc/(max70C-min30C)]))
# ss = np.stack(ss, 0).reshape(1000, 3)

def line(p,x):
    k,b=p
    return k*x+b
def leastl(p,x,y):
    return line(p,x)-y
def weightequal(x1,x2,x3):
    x=np.clip(x1/x2,0,0.99571227)
    return np.log((0.99862891-x)/1.01976519)/(-1.14557532)*x2*(-0.94674696*x3+1)

def aaical(label_seeds, rgt,rgtt, ilist,results):
    aat=np.zeros((len(ilist),rgt.shape[0],18))
    for i,_ in enumerate(ilist):
        mask = (label_seeds == ilist[i]).astype(np.float32).reshape(-1)
        if mask.sum() < 50:
            aat[i,:,:]== np.nan
        else:
            rgti = rgt.reshape((rgt.shape[0], -1))[:, np.where(mask)][:, 0, :]
            rgtti= rgtt.reshape((rgt.shape[0], -1))[:, np.where(mask)][:, 0, :]
            mins = np.percentile(rgtti, 20, -1)
            maxs = np.percentile(rgtti, 80, -1)
            NN=len(rgti[0])
            idxs=np.argsort(rgtti,-1)[:,NN//5:NN-NN//5]
            masks=np.zeros_like(rgti)
            for ii in range(rgt.shape[0]):
                masks[ii, idxs[ii]] = 1
            masks=masks.astype(np.bool_)
            ranges=maxs-mins
            aat[i, :, 2] = ranges
            aat[i,:,16]=np.percentile(rgtti, 60, -1)-np.percentile(rgtti, 40, -1)
            rgtir=(rgtti-mins[:,None])/(maxs[:,None]-mins[:,None])
            for j in range(0, rgt.shape[0]):
                maskdif1 = (masks[j])  & (masks[(j - 1) % rgt.shape[0]])
                maskdif2 = (masks[j]) & (masks[(j + 1) % rgt.shape[0]])
                dif=0
                meann=0
                count=0
                if maskdif1.astype(np.int16).sum()>15:
                    ps,succ=leastsq(leastl,(1,0),(rgtir[(j - 1) % rgt.shape[0]][maskdif1],rgtir[j][maskdif1]))
                    if succ>3:
                        ps=[0,0]
                    dif1 = rgtir[j] - (ps[0]*rgtir[(j - 1) % rgt.shape[0]]+ps[1])
                    dif1 = (((dif1[maskdif1]))).std(ddof=1) ** 2*ranges[j]**2
                    difall=rgtti[(j)][maskdif1].std(ddof=1) ** 2

                    aat[i, j, 5]=dif1
                    aat[i,j,7]=rgtti[(j - 1) % rgt.shape[0]][maskdif1].std(ddof=1) ** 2
                    aat[i, j, 8] = difall
                    aat[i,j,11]=rgtti[(j)][maskdif1].mean()
                    dif=dif+weightequal(aat[i, j, 5],aat[i, j, 8],aat[i, j, 2])
                    meann=meann+aat[i,j,11]
                    count=count+1
                    zeropercent = ((rgti[(j)][maskdif1] > 1/4096)&(rgti[(j - 1) % rgt.shape[0]][maskdif1] > 1/4096)).mean()
                    aat[i, j, 4] = zeropercent
                    onespercent=((rgti[(j)][maskdif1] < 0.99)&(rgti[(j - 1) % rgt.shape[0]][maskdif1] < 0.99)).mean()
                    aat[i, j, 13] = onespercent
                if maskdif2.astype(np.int16).sum()>15:
                    ps, succ = leastsq(leastl, (1, 0), (rgtir[(j + 1) % rgt.shape[0]][maskdif2], rgtir[j][maskdif2]))
                    if succ>3:
                        ps=[0,0]
                    dif2 = rgtir[j] - (ps[0] * rgtir[(j + 1) % rgt.shape[0]] + ps[1])
                    dif2 = (((dif2[maskdif2]))).std(ddof=1) ** 2*ranges[j]**2
                    difall=rgtti[(j)][maskdif2].std(ddof=1) ** 2

                    aat[i, j, 6]=dif2
                    aat[i,j,9]=rgtti[(j + 1) % rgt.shape[0]][maskdif2].std(ddof=1) ** 2
                    aat[i, j, 10] = difall
                    aat[i,j,12]=rgtti[(j)][maskdif2].mean()
                    dif = dif + weightequal(aat[i, j, 6],aat[i, j, 10],aat[i, j, 2])
                    meann = meann + aat[i, j, 12]
                    count = count + 1
                    zeropercent = ((rgti[(j)][maskdif2] > 1/4096) & (rgti[(j + 1) % rgt.shape[0]][maskdif2] > 1/4096)).mean()
                    aat[i, j, 14] = zeropercent
                    onespercent = ((rgti[(j)][maskdif2] < 0.99) & (rgti[(j + 1) % rgt.shape[0]][maskdif2] < 0.99)).mean()
                    aat[i, j, 15] = onespercent
                aat[i, j, 1] = meann/count if count else 0 # *ff(zeropercent)
                aat[i, j, 0] = dif / count if count else 0
            aat[i,:, -1] = i
    aat[:, 0, -1] = ilist
    results.extend(aat)




if __name__ == '__main__':


    aas=[]
    namelist=[]
    for imgidx in range(54,60):#[55]:#(list(range(11,51))+list(range(56,60))):#,1,2,3,4,6,7,8,9,10,51,52,53,54,55]:
        namelist.append('in'+str(imgidx))
        namelist.append('gt' + str(imgidx))
    #namelist=[ 'Urban','salinas','paviaU',]
    #namelist=['lehavim_0910-1607']
    #namelist=glob.glob('E:\Datas\mats\\*\\*.mat')

    # for name in namelist:
    #     rgt=np.array(h5py.File(name)['rad']);rgt=rgt/4096;name='E:\Datas\\aa\\'+name[16:-4]
    #     C,H,W=rgt.shape
    #     cut=280
    #     stepH=H//cut
    #     aa=np.array([0,H-stepH*cut])
    #     stepW=W//cut
    #     bb=np.array([0,W-stepW*cut])
    #     for i in range(len(aa)):
    #         for j in range(len(bb)):
    #             np.save('E:\\Datas\\bb\\'+ name[12:] + str(100 + i) + str(100 + j) + 'allgt.npy',
    #                     rgt[:, int(aa[i]):int(aa[i])+stepH*cut:stepH, int(bb[j]):int(bb[j])+stepW*cut:stepW])
    for name in namelist:
        rgt = np.load('F:\SERT-master\models\competing_methods\\rgr'+name+'.npy') /4096
        #rgt=scipy.io.loadmat('F:\Hyperspectral-Classification-master\Datasets\\allCHW\\'+name+'.mat')['rad']
        #rgt=np.array(h5py.File(name)['rad']);rgt=rgt/4096;name='E:\Datas\\aa\\'+name[16:-4]

        C,H,W=rgt.shape
        cc=C//3
        img=np.stack([rgt[0:cc].mean(0),rgt[cc:2*cc].mean(0),rgt[cc*2:cc*3].mean(0)],2).astype(np.float32)
        img=img/img.max()
        seeds = cv2.ximgproc.createSuperpixelSEEDS(img.shape[1],img.shape[0],img.shape[2],int(img.shape[1]*img.shape[0]//(400)),15,5,9,True)
        seeds.iterate(img,20)  #输入图像大小必须与初始化形状相同，迭代次数为10
        mask_seeds = seeds.getLabelContourMask()
        label_seeds = seeds.getLabels()
        number_seeds = seeds.getNumberOfSuperpixels()
        aa=np.zeros((number_seeds,len(rgt),17))
        NN=[]
        lists=np.unique(label_seeds)
        ks=[]
        bs=[]
        # rgtmeans=np.zeros((len(lists),len(ll),2))
        # rgtmeans[:, :, 0]=lists[:,None]
        jstep = []
        for ri in rgt.reshape(rgt.shape[0], -1):
            temp = np.unique(ri)
            temp = temp[temp > (1 / 2 ** 16)]
            jstep.append(temp[0])
        jstep = np.array(jstep)
        rgtt = (np.random.rand(rgt.shape[0], rgt.shape[1], rgt.shape[2]) - 0.5) * jstep[:, None, None] + rgt

        meanses=[]
        stdses=[]
        try:
            processes = 4
            pool = Pool(processes=processes)
            timest = time.time()
            manager = Manager()
            results = manager.list()
            ises = {}
            for i in range(processes):
                ises[str(i)] = lists[i::processes]
            for i in range(processes):
                pool.apply_async(aaical, (label_seeds, rgt,rgtt, ises[str(i)], results))

            pool.close()
            pool.join()
            ded = np.array(results)
            aa[(ded[:, 0, -1]).astype(np.int64)] = ded[:, :, :-1]
            aa[:, :, 3] = aa[:, :, 1] * 0+ 1  # ((aa[:,:,1]*(maxr-minr)+minr)* kt + bt)/((maxr-minr)**2)
            del pool, ded, results, seeds
            print(time.time() - timest)
        except:
            continue
        aa[np.where(aa==0)]=np.nan

        order=18

        x = np.percentile(rgt.reshape(rgt.shape[0], -1), 97, -1)
        peaks = scipy.signal.argrelextrema(-x, np.greater, order=order)
        peaks = peaks[0]
        if len(peaks) == 0:
            peaks = np.array([0, len(x) - 1])
        if peaks[0] < 10:
            peaks[0] = 0
        else:
            peaks = np.concatenate([np.zeros(1, ).astype(np.int64), peaks])
        if peaks[-1] > (len(x) - 10):
            peaks[-1] = len(x)
        else:
            peaks = np.concatenate([peaks, (len(x)) * np.ones(1, ).astype(np.int64)])
        temp = []
        for j in range(len(peaks) - 1):
            step = peaks[j + 1] - peaks[j]
            if step > order*2:
                num = step // order
                temp.append(peaks[j] + np.round(np.arange(1, num) * step / num))
        peaks = np.concatenate([peaks] + temp)
        peaks.sort()
        peaks=peaks.astype(np.int64)

        #plt.plot(x)
        #plt.plot(peaks, x[peaks], "o")

        kbs=np.zeros((len(rgt),4))

        for j in range(len(peaks)-1):
            st = peaks[j]
            ed = peaks[j + 1]
            nanstd = (aa[:, st:ed, :].min(-1) > 1e-20) & (aa[:, st:ed, 4] > 0.99) & (aa[:, st:ed, 13] > 0.99) & (
                        aa[:, st:ed, 14] > 0.99) & (aa[:, st:ed, 15] > 0.99)
            nanstd = nanstd & (((aa[:, st:ed, 8]) / (aa[:, st:ed, 2] ** 2)) > 0.07053223651129881) & (
                    ((aa[:, st:ed, 8]) / (aa[:, st:ed, 2] ** 2)) < 0.07853223651129881)
            nanstd = nanstd & (((aa[:, st:ed, 10]) / (aa[:, st:ed, 2] ** 2)) > 0.07053223651129881) & (
                    ((aa[:, st:ed, 10]) / (aa[:, st:ed, 2] ** 2)) < 0.07853223651129881)
            stds = aa[:, st:ed, 0][np.where(nanstd)]
            means = aa[:, st:ed, 1][np.where(nanstd)]

            weights = 1
            idxs = np.argsort(means)
            means = means[idxs]
            stds = stds[idxs]
            NNi=3
            number=len(means)//NNi
            means=np.array([np.median(means[ij*NNi:(ij+1)*NNi]) for ij in range(number)])
            stds=np.array([np.median(stds[ij*NNi:(ij+1)*NNi]) for ij in range(number)])

            k = (np.percentile(stds, 75) - np.percentile(stds, 25)) / (
                        np.percentile(means, 75) - np.percentile(means, 25))
            b = np.percentile(stds, 50) - np.percentile(means, 50) * k
            thita, succ = leastsq(leastl, ((k, b)), (means, stds))
            k = thita[0]
            b = thita[1]
            if k < 0:
                k = 0
                b = np.median(stds)
            x = np.array([means.min(), means.max()])
            y = x * k + b
            kbs[st:ed, 0] = k
            kbs[st:ed, 1] = b
            diff = (((means * k + b) - stds))
            poserros = np.percentile(diff[diff > 0], 90)
            negerros = np.percentile(-diff[diff < 0], 90)
            kbs[st:ed, 2] = poserros
            kbs[st:ed, 3] = negerros
            fig = plt.figure(figsize=(6, 4), dpi=600)
            plt.cla()
            plt.clf()
            axes = fig.add_subplot(1, 1, 1)
            kk = "{:.2e}".format(k)
            bb = "{:+.2e}".format(b)
            idx = np.arange(len(stds))
            np.random.shuffle(idx)
            means = means[idx[:2000]]
            stds = stds[idx[:2000]]
            p2 = axes.scatter(means, stds, s=1, marker='.', c='b', )
            p2 = axes.scatter(-1, -1, s=10, marker='.', c='b', )
            p3, = axes.plot(means, means * k + b, 'k')
            dif = np.abs(stds - (means * k + b))
            font = {'family': 'Times New Roman', 'size': 13, }
            axes.legend([p2, p3], ['Variance', 'fitted_line'], loc="upper left", prop=font)
            # p3,"{:.3} * Intensity +{:.6}".format(k,b)
            plt.ylabel("Variance", fontdict=font)
            plt.xlabel("Intensity", fontdict=font)
            xlim = np.percentile(means, 99) * 1.2
            ylim = (xlim * k + b) + 4 * poserros
            axes.text(xlim * 0.55, ylim * 0.05, kk + r"$\times$" + "Intensity" + bb, fontsize=12,
                      family='Times New Roman')
            plt.xlim((0, xlim))
            plt.ylim((0, ylim))
            plt.savefig('linefigrg117{}_{}.png'.format(name, peaks[j]))

            plt.close()

        # for j in  range(aa.shape[1]):
        #     nanstd = np.isnan(aa[:, j, 0])|np.isnan(aa[:, j, 1])
        #     stds = aa[:, j, 0][np.where(~nanstd)]
        #     weights = 1.09
        #     means = aa[:, j, 1][np.where(~nanstd)]
        #     stds = stds / weights
        #     if len(stds) <=10:
        #         continue
        #     diff=(((means*kbs[j, 0]+kbs[j, 1])-stds))
        #     negdiff=-diff[diff < 0]
        #     posdiff=diff[diff > 0]
        #
        #     if len(posdiff)<=10:
        #         scorep=True
        #     elif np.percentile(posdiff, 90)<(kbs[j, 2]*1.5):
        #         scorep=True
        #     else:
        #         scorep=False
        #     if len(negdiff)<=10:
        #         scoren=True
        #     elif np.percentile(negdiff, 90)<(kbs[j, 3]*1.5):
        #         scoren=True
        #     else:
        #         scoren=False
        #     if scoren&scorep:
        #         continue
        #     else:
        #         score=sklearn.metrics.r2_score(stds,means * k + b )
        #         if score>0:#R2为负数就是你得到的拟合函数预测误差大于Y=平均值这条函数的预测误差
        #             continue
        #     kbs[j, 0] = 0
        #     kbs[j, 1] = np.median(stds)
        #     try:
        #         model_ransac = linear_model.RANSACRegressor()
        #         model_ransac.fit(means.reshape(-1, 1), stds,3-means)
        #         k = model_ransac.estimator_.coef_[0]
        #         b = model_ransac.estimator_.intercept_
        #         if k>0:
        #             kbs[j, 0]=k
        #             kbs[j, 1]=b
        #         else:
        #             continue
        #     except:
        #         continue
        np.save('ksbs'+name+'.npy',kbs)
        np.save('peaks'+name+'.npy',peaks)
        np.save('aa' + name + '.npy', aa)
        a=0
    # np.save('F:\SERT-master\models\\' +name  + 'k.npy',k)
    # np.save('F:\SERT-master\models\\' + name + 'b.npy', b)

    # aas.append(aa.copy())
    # means=np.concatenate(meanses)
    # stds=np.concatenate(stdses)
    # a=0
    # np.save(name+'means',means)
    # np.save(name + 'stds.npy', stds)
    #

    # #
    # # np.save(name+'k.npy',k)
    # # np.save(name + 's3.npy', b)
    # idx = np.ones(label_seeds.max() + 1) * (-1)
    # idx[rgtmeans[:, 15, 0].astype(np.int32)] = np.arange(len(rgtmeans[:, 15, 0]))
    # mask = idx.astype(np.int32)[label_seeds]
    # # np.save(name+'mask_seeds.npy',mask_seeds)
    # # np.save(name+'label_seeds.npy',mask)
    # # np.save(name+'ll.npy',ll)
    # rmeans=rgtmeans[:,:,1][mask].transpose((2,0,1)).astype(np.float32)
    # np.save(name+'rmeans.npy',rmeans)
    # # #np.save(name+'rgtreg.npy',rgtreg.reshape(rgt.shape[0],rgt.shape[1],rgt.shape[2]))
    # # print(name+' {:.3} {:.3}'.format(k,b))
    #
    # a=0
    # rmeans=np.load(name+'rmeans.npy')
    # C,H,W=rgt.shape
    # if not (rgt.shape==rmeans.shape):
    #     continue
    # cut=280
    # step=280
    # aa=H//step
    # aa=np.arange(aa)*step+(H-step*aa)//2
    # aa=aa[:-1]
    # bb = W // step
    # bb = np.arange(bb) *step + (W - step*bb) // 2
    # bb = bb[:-1]
    #
    # for i in range(len(aa)):
    #     for j in range(len(bb)):
    #         np.save('E:\\Datas\\bb\\'+ name[12:] + str(100 + i) + str(100 + j) + 'gt.npy',
    #                 rgt[:, int(aa[i]):int(aa[i]) + cut, int(bb[j]):int(bb[j]) + cut])
    #         np.save('E:\\Datas\\bb\\' + name[12:] + str(100 + i) + str(100 + j) + 'rmeans.npy',
    #                 rmeans[:, int(aa[i]):int(aa[i]) + cut, int(bb[j]):int(bb[j]) + cut])
    # a=0



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import leastsq
# def Fun(p,x):                        # 定义拟合函数形式
#     a1,a2 = p
#     return a1*x+a2
# def error (p,x,y):                   # 拟合残差
#     return (Fun(p,x)-y)
# ss=[]
# # for nb in np.arange(2)/1:
# #     na=1/()
# #     nnc =1/np.sqrt(1+na**2)
# #     nna=nnc*na
# #     nnb=nnc*nb
# #     A=np.random.randn(10000,)*nna
# #     B=A+np.random.randn(10000,)*nnb
# #     C=A+np.random.randn(10000,)*nnc
# #     min30C=np.percentile(C,30)
# #     max70C=np.percentile(C,70)
# #     maskdif=(C>=min30C)&(C<=max70C)
# #     tt=leastsq(error, (1, 1), (B[maskdif], C[maskdif]))[0]
# #     tt=C-(tt[0]*B+tt[1])
# #     db=(tt[0]*B+tt[1])[maskdif].std()**2
# #     dc=C[maskdif].std()**2
# #     de=tt[maskdif].std()**2
# #     #plt.plot(na,tt[maskdif].std()**2-db,'.')
# #     ss.append(np.array([db,dc,de,nnc]))
# # ss=np.stack(ss,0).reshape(2,20000,4)
# def Fun(p,x):
#     a1,a3,a5,a7= p
#     return a1
#
#
# tt_1, succ = leastsq(error, (1, 1, 1, 1), (ss[-1, :, 2] / ss[-1, :, 1], (ss[-1, :, 3])))