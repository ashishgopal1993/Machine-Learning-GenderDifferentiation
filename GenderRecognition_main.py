#train_models.py

import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GMM
import python_speech_features as mf
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")


def get_MFCC(sr,audio):
    features = mf.mfcc(audio,sr, 0.025, 0.01, 13,appendEnergy = False)
    features = preprocessing.scale(features)
    return features


# path to training data
source = r"F:\Speaker Recognition\Research\From Applied Machine Learning\Stunt\Training_Data\Speaker2"
# path to save trained model
destination = "F:\Speaker Recognition\Research\From Applied Machine Learning\Stunt\Models"
files = [os.path.join(source, f) for f in os.listdir(source) if f.endswith('.wav')]
features = np.asarray(())

for f in files:
    sr, audio = read(f)
    vector = get_MFCC(sr,audio)
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

gmm = GMM(n_components=8, n_iter=200, covariance_type='diag', n_init = 3)
gmm.fit(features)
picklefile = f.split("\\")[-2].split(".wav")[0]+".gmm"

# model saved as .gmm
cPickle.dump(gmm, open(destination + picklefile, 'wb'))
print('modeling completed for gender:',picklefile)