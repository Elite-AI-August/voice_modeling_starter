# import libraires
import os, librosa, json, datetime, time
import numpy as np
from tqdm import tqdm

def get_wav():
	listdir=os.listdir()
	wavfiles=list()
	for i in range(len(listdir)):
		if listdir[i].endswith('.wav'):
			wavfiles.append(listdir[i])
	return wavfiles

# get statistical features in numpy
def stats(matrix):
    mean=np.mean(matrix)
    std=np.std(matrix)
    maxv=np.amax(matrix)
    minv=np.amin(matrix)
    median=np.median(matrix)

    output=np.array([mean,std,maxv,minv,median])
    
    return output

# get labels for later 
def stats_labels(label, sample_list):
    mean=label+'_mean'
    std=label+'_std'
    maxv=label+'_maxv'
    minv=label+'_minv'
    median=label+'_median'
    sample_list.append(mean)
    sample_list.append(std)
    sample_list.append(maxv)
    sample_list.append(minv)
    sample_list.append(median)

    return sample_list

def featurize_audio(filename):
   # if categorize == True, output feature categories 
    print('librosa featurizing: %s'%(filename))

    # initialize lists 
    onset_labels=list()
    y, sr = librosa.load(filename)

    # FEATURE EXTRACTION
    ######################################################
    # extract major features using librosa
    mfcc=librosa.feature.mfcc(y)
    poly_features=librosa.feature.poly_features(y)
    chroma_cens=librosa.feature.chroma_cens(y)
    chroma_cqt=librosa.feature.chroma_cqt(y)
    chroma_stft=librosa.feature.chroma_stft(y)
    tempogram=librosa.feature.tempogram(y)

    spectral_centroid=librosa.feature.spectral_centroid(y)[0]
    spectral_bandwidth=librosa.feature.spectral_bandwidth(y)[0]
    spectral_contrast=librosa.feature.spectral_contrast(y)[0]
    spectral_flatness=librosa.feature.spectral_flatness(y)[0]
    spectral_rolloff=librosa.feature.spectral_rolloff(y)[0]
    onset=librosa.onset.onset_detect(y)
    onset=np.append(len(onset),stats(onset))
    # append labels 
    onset_labels.append('onset_length')
    onset_labels=stats_labels('onset_detect', onset_labels)

    tempo=librosa.beat.tempo(y)[0]
    onset_features=np.append(onset,tempo)

    # append labels
    onset_labels.append('tempo')

    onset_strength=librosa.onset.onset_strength(y)
    onset_labels=stats_labels('onset_strength', onset_labels)
    zero_crossings=librosa.feature.zero_crossing_rate(y)[0]
    rmse=librosa.feature.rmse(y)[0]

    # FEATURE CLEANING 
    ######################################################

    # onset detection features
    onset_features=np.append(onset_features,stats(onset_strength))


    # rhythm features (384) - take the first 13
    rhythm_features=np.concatenate(np.array([stats(tempogram[0]),
                                      stats(tempogram[1]),
                                      stats(tempogram[2]),
                                      stats(tempogram[3]),
                                      stats(tempogram[4]),
                                      stats(tempogram[5]),
                                      stats(tempogram[6]),
                                      stats(tempogram[7]),
                                      stats(tempogram[8]),
                                      stats(tempogram[9]),
                                      stats(tempogram[10]),
                                      stats(tempogram[11]),
                                      stats(tempogram[12])]))
    rhythm_labels=list()
    for i in range(13):
        rhythm_labels=stats_labels('rhythm_'+str(i), rhythm_labels)

    # spectral features (first 13 mfccs)
    spectral_features=np.concatenate(np.array([stats(mfcc[0]),
                                        stats(mfcc[1]),
                                        stats(mfcc[2]),
                                        stats(mfcc[3]),
                                        stats(mfcc[4]),
                                        stats(mfcc[5]),
                                        stats(mfcc[6]),
                                        stats(mfcc[7]),
                                        stats(mfcc[8]),
                                        stats(mfcc[9]),
                                        stats(mfcc[10]),
                                        stats(mfcc[11]),
                                        stats(mfcc[12]),
                                        stats(poly_features[0]),
                                        stats(poly_features[1]),
                                        stats(spectral_centroid),
                                        stats(spectral_bandwidth),
                                        stats(spectral_contrast),
                                        stats(spectral_flatness),
                                        stats(spectral_rolloff)])) 

    spectral_labels=list()
    for i in range(13):
        spectral_labels=stats_labels('mfcc_'+str(i), spectral_labels)
    for i in range(2):
        spectral_labels=stats_labels('poly_'+str(i), spectral_labels)
    spectral_labels=stats_labels('spectral_centroid', spectral_labels)
    spectral_labels=stats_labels('spectral_bandwidth', spectral_labels)
    spectral_labels=stats_labels('spectral_contrast', spectral_labels)
    spectral_labels=stats_labels('spectral_flatness', spectral_labels)
    spectral_labels=stats_labels('spectral_rolloff', spectral_labels)

    # power features
    power_features=np.concatenate(np.array([stats(zero_crossings),
                                         stats(rmse)]))
    power_labels=list()
    power_labels=stats_labels('zero_crossings',power_labels)
    power_labels=stats_labels('RMSE', power_labels) 

    # can output numpy array of everything if we don't need categorizations 
    features = np.concatenate(np.array([onset_features,
                                   rhythm_features,
                                   spectral_features,
                                   power_features]))
    labels=onset_labels+rhythm_labels+spectral_labels+power_labels

    return features, labels

def featurize_folder(curdir, folder):
    os.chdir(curdir)
    os.chdir(folder)
    wavfiles=get_wav()
    features_model=list()
    class_labels=list()
    listdir=os.listdir()
    for i in tqdm(range(len(wavfiles)), desc=folder):
        jsonfilename=wavfiles[i][0:-4]+'.json'

        try:
            if jsonfilename not in listdir:
                features, labels = featurize_audio(wavfiles[i])

                data={'features':dict(zip(labels,features.tolist())),
                      'wavfile': wavfiles[i],
                      'datetime': str(datetime.datetime.now())}

                
                jsonfile=open(jsonfilename, 'w')
                json.dump(data,jsonfile)
                jsonfile.close()
                
                features_model.append(features)
                class_labels.append(folder)
            else:
                print('SKIPPING - already featurized %s'%(wavfiles[i]))
                print(listdir[i])
                g=json.load(open(wavfiles[i][0:-4]+'.json'))
                temp=g['features']
                features_model.append(list(temp.values()))
                class_labels.append(folder)
        except:
            pass

    return features_model, class_labels

def make_prediction(model):
    listdir=os.listdir()
    for i in range(len(listdir)):
        if listdir[i].endswith('.json'):
           # try:
            g=json.load(open(listdir[i]))
            prediction=clf.predict(np.array(list(g['features'].values())).reshape(1,-1))
            g['prediction']=prediction
            jsonfile=open(listdir[i],'w')
            json.dump(g,jsonfile)
            jsonfile.close()
            # except:
                # print('error')

############################################################
# get back all the features and their relative class so that we can model.
############################################################
curdir=os.getcwd()

features_model_males, class_labels_males=featurize_folder(curdir, 'males')
features_model_females, class_labels_females=featurize_folder(curdir, 'females')
features_model=features_model_males + features_model_females
class_labels=class_labels_males + class_labels_females
# print(features_model)
# print(class_labels)

# now model the data in some way 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create training data
X_train, X_test, y_train, y_test = train_test_split(features_model, class_labels, test_size=0.20, random_state=42)

# initialize machine learning mode
clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf = LogisticRegression(random_state=0)

# train a machine model 
print('fitting model')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

# evaluate accuracy 
accuracy=accuracy_score(y_test, y_pred)
print(accuracy)

# # sample prediction
# print(np.array(X_test[1]).shape)
# print(clf.predict(np.array(X_test[1]).reshape(1,-1)))

# # make a prediction of all the models and put them in a spreadsheet
# os.chdir(curdir)
# os.chdir('males')
# make_prediction(clf)
