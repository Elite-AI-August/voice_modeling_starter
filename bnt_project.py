import os, time
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

def featurize(audioclip):
    #read audioclip
    print(audioclip)
    y, sr = librosa.load(audioclip)

    #mfcc (short term power spectrum of a sound)
    features=list()
    labels=list()

    # mfcc features 
    mfcc=librosa.feature.mfcc(y=y, sr=sr)[0]
    mfcc_avg=np.average(mfcc[0])
    mfcc_max=np.amax(mfcc[0])
    mfcc_min=np.amin(mfcc[0])
    mfcc_std=np.std(mfcc[0])
    features.append(mfcc_avg)
    features.append(mfcc_max)
    features.append(mfcc_min)
    features.append(mfcc_std)
    labels.append('mfcc average')
    labels.append('mfcc max')
    labels.append('mfcc min')
    labels.append('mfcc std')

    # spectral rolloff 
    sr_=librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    sr_avg=np.average(sr_[0])
    sr_max=np.amax(sr_[0])
    sr_min=np.amin(sr_[0])
    sr_std=np.std(sr_[0])
    features.append(sr_avg)
    features.append(sr_max)
    features.append(sr_min)
    features.append(sr_std)
    labels.append('spectral rolloff average')
    labels.append('spectral rolloff maximum')
    labels.append('spectral rolloff minimum')
    labels.append('spectral rolloff std')

    # chromagram
    chroma_stft=librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_avg=np.average(chroma_stft[0])
    chroma_stft_max=np.amax(chroma_stft[0])
    chroma_stft_min=np.amin(chroma_stft[0])
    features.append(chroma_stft_avg)
    features.append(chroma_stft_max)
    features.append(chroma_stft_min)
    labels.append('chroma stft average')
    labels.append('chroma stft maximum')
    labels.append('chroma stft minimum')

    sc=librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sc_avg=np.average(sc[0])
    sc_max=np.amax(sc[0])
    sc_min=np.amin(sc[0])
    sc_std=np.std(sc[0])
    features.append(sc_avg)
    features.append(sc_max)
    features.append(sc_min)
    features.append(sc_std)
    labels.append('sc average')
    labels.append('sc max')
    labels.append('sc min')
    labels.append('sc std')

    dictionary=dict(zip(labels,features))
    print(dictionary)
    # print(list(dictionary)) # this is = to labels
    # print(labels)
    # print(list(dictionary.values())) # this is = to features
    # print(features)

    return features, labels 

def featurize_by_folder(class_):
    curdir=os.getcwd()
    os.chdir(class_)
    listdir=os.listdir()
    features=list()
    labels=list()

    for i in tqdm(range(len(listdir)), desc=class_):
        if listdir[i].endswith('.wav'):
            audiofile=listdir[i]
            features_mfcc, labels_mfcc = featurize(audiofile)
            features.append(features_mfcc)
            labels.append(class_)
    os.chdir(curdir)

    return features, labels 
# list of features / go into a folder with audio files 

features_males, labels_males = featurize_by_folder('males')
features_females, labels_females = featurize_by_folder('females')
features = features_males + features_females
labels = labels_males + labels_females 

# features --> [[5,2,4,5], [3,2,4,5], [2,3,4,5],....,[2,3,4,5]]
# labels --> ['males','males','males',.....,'females']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=25)
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
accuracy_mfcc=clf.score(X_test, y_test)
print(accuracy)
