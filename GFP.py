## author:zhaoshuzhi
# header files
import os
import numpy as np
import scipy.io as scio
from sampen import sampen2
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten
import scipy.cluster.hierarchy as sch
import mne


# load data .mat
data_dir = '/media/lhj/Momery/PD_GCN/Script/test_ChebNet/data2'
files = os.listdir(data_dir)
xT = []
wins = [10,20,25]
biosemi_montage = mne.channels.make_standard_montage('biosemi64')
# info
info = mne.create_info(ch_names=biosemi_montage.ch_names,sfreq=250.,ch_types='eeg')
for winSize in wins:
    for i in range(0,700,winSize):
        xT.append(i+winSize/2)
    for file in files: 
        print("++++++++++++++++++++++ The process of"+file+" ++++++++++++++++++++++")
        data=scio.loadmat(data_dir+'/'+file)  
        ll = list((data.keys()))
        ERPs = []
    #     print(data)
        for i in ll:
            if 'Category' in i:
                ERPs.append(data[i])
    #             print(np.array(data[i]).shape)
        ERPs = np.mean(ERPs,axis=0) 
        mCERPs = np.mean(ERPs,axis=0)
        print('one subject mean RRPs:\n',np.array(ERPs).shape,'\none subject mean mCRRPs:\n',np.array(mCERPs).shape)
        # Global Field Potential
        tGFPs = []
    #     for ii in range(np.array(ERPs).shape[0]):
    #         GFPs = []
    #         for jj in range(0,np.array(ERPs).shape[1],winSize):
    # #             print(ERPs[ii][jj:jj+winSize])
    #             SE = sampen2(ERPs[ii][jj:jj+winSize])
    #             GFP = (SE[0][1]+SE[1][1]+SE[2][1])/3
    #             GFPs.append(GFP)
    #         tGFPs.append(GFPs)
    #     tGFPs = np.mean(tGFPs,axis=0)    
        # Cluster timeserial in mean channels
        features = []
        for ii in range(0,np.array(mCERPs).shape[0],winSize):
            features.append(mCERPs[ii:ii+winSize])
            SE = sampen2(mCERPs[ii:ii+winSize])
            GFP = (SE[0][1]+SE[1][1]+SE[2][1])/3
            tGFPs.append(GFP)
    #     features = np.array(features).transpose().tolist()
        whitened = whiten(features)
        # hierarchy
        disMat = sch.distance.pdist(features,'euclidean')
        Z = sch.linkage(disMat,method='average')
        cluster = sch.fcluster(Z,t=1, criterion='inconsistent')
        centroid = kmeans(whitened,4)[0]
        label = vq(whitened,centroid)[0]
    #     book = np.array((whitened[0],whitened[2]))
        print('features shape:',np.array(features).shape,'cluster shape:',centroid[0].shape,'label shape:',label.shape)
        print('data shape:',whitened.shape,' unique label:',np.unique(label),'label:\n',label)
        tGFPsp = [[] for i in np.unique(label)]
        for cl in np.unique(label):
            mask_data = tGFPs*(label==cl).astype(int)
    #         print('mask:',(label==cl).astype(int),'\nmask data',mask_data)
            tGFPsp[cl].append(mask_data)
            # plot topomap
    #         data1 = []
    #         temp = [val for val in (label==cl).astype(int) for ii in range(winSize)]
    #         print('temp shape:',np.array(temp).shape)
    #         data1.append([ii*temp for ii in ERPs])
    #         data1 = np.array(data1).reshape(65,700)[0:64,:]
            temp = [val for val in label==cl for ii in range(winSize)]
            data1 = ERPs[1:65,temp]
            print('data1 shape:',np.array(data1).shape)
            evoked = mne.EvokedArray(data1,info)
            evoked.set_montage(biosemi_montage)
            mne.viz.plot_topomap(evoked.data[:,0],evoked.info,show=False)
            plt.savefig('/media/lhj/Momery/PD_GCN/result/topoMap/'+file.replace('.mat','_cluster_')+str(cl)+'_'+str(winSize)+'_.tif')
            plt.close('all')
    #         plt.show()

    #     print('mask GFP:',tGFPsp)

        # plot GFP
#         plt.stackplot(xT, tGFPsp[0],tGFPsp[1],tGFPsp[2],tGFPsp[3])
    #     plt.savefig('/media/lhj/Momery/PD_GCN/result/GFP/'+file.replace('.mat','')+'_1_10w.tif')
#         plt.close('all')


    #     plt.show()