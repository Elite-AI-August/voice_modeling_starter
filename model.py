import os
import numpy as np
import librosa, pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

def featurize(audiofile):

	# read the audio file 
	y, sr = librosa.load(audiofile)

	# featurize the audio file
	# http://librosa.github.io/librosa/feature.html
	features_ =librosa.feature.mfcc(y=y, sr=sr)[0]

	# calculate features
	features=list()
	mfcc_1_avg = np.average(features_[0])
	mfcc_1_std = np.std(features_[0])
	mfcc_1_median = np.median(features_[0])
	features.append(mfcc_1_avg)
	features.append(mfcc_1_std)
	features.append(mfcc_1_median)

	# get labels 
	labels=list()
	labels.append('mfcc_1_average')
	labels.append('mfcc_1_std')
	labels.append('mfcc_1_median')

	# return features and labels 
	return features, labels 

# def model():
# audiofile='test.wav'
# features, labels = featurize(audiofile)
# print(features)
# print(labels)

features=list()
labels=list()

# base directory
curdir=os.getcwd()

os.chdir('males')
listdir=os.listdir()

for i in tqdm(range(len(listdir)), desc='males'):
	if listdir[i].endswith('.wav'):
		print('featurizing %s'%(listdir[i].upper()))
		features_, labels_ = featurize(listdir[i])
		features.append(features_)
		labels.append('males')

# go back to directory
os.chdir(curdir)
os.chdir('females')
listdir=os.listdir()

for i in tqdm(range(len(listdir)), desc='females'):
	if listdir[i].endswith('.wav'):
		print('featurizing %s'%(listdir[i].upper()))
		features_, labels_ = featurize(listdir[i])
		features.append(features_)
		labels.append('females')

print(features)
print(labels)

# build a machine learning model
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
os.chdir(curdir)
print('training model')
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
accuracy=clf.score(X_test, y_test)
print('accuracy--> ')
print(accuracy)

# save a model 
print('saving model')
modelfile=open('model.pickle','wb')
pickle.dump(clf,modelfile)
modelfile.close()

# load the model 
print('loading model')
model=pickle.load(open('model.pickle','rb'))
accuracy2=model.score(X_test,y_test)
print('loaded model accuracy-->')
print(accuracy2)