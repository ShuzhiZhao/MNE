import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

biosemi_montage = mne.channels.make_standard_montage('biosemi64')
# data
data = np.random.randn(64,10)
# info
info = mne.create_info(ch_names=biosemi_montage.ch_names,sfreq=250.,ch_types='eeg')
# evokeds
evoked = mne.EvokedArray(data,info)
# evokeds channels
evoked.set_montage(biosemi_montage)
# plot
mne.viz.plot_topomap(evoked.data[:,0],evoked.info,show=False)
plt.show()