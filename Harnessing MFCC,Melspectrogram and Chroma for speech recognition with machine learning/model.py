import glob
import os
import pickle

import librosa
import time
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

tess_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']


def extract_mel_spectrogram(file_name):
    # Load audio file using librosa
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')

    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128)

    # Calculate the mean across all frames
    mean_mel_spectrogram = np.mean(mel_spectrogram.T, axis=0)

    return mean_mel_spectrogram


def extract_chroma(file_name):
    # Load audio file using librosa
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')

    # Compute the short-time Fourier transform (STFT) of the input signal
    stft = np.abs(librosa.stft(X))

    # Compute the chromagram
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)

    # Calculate the mean across all frames
    mean_chroma = np.mean(chroma.T, axis=0)

    return mean_chroma


def extract_mfcc(file_name):
    # Load audio file using librosa
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')

    # Compute the MFCCs

    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)

    # Calculate the mean across all frames
    mean_mfccs = np.mean(mfccs.T, axis=0)

    return mean_mfccs
def extract_feature(file_name):

    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')
    result = np.array([])
    print(result)

    stft = np.abs(librosa.stft(X))
    chromas = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chromas))
    print(result)

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs))
    print(result)

    mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128).T, axis=0)
    result = np.hstack((result, mels))
    print(result)

    return result


def load_data():
    sound, emo = [], []

    for file in glob.glob(
            "TESS Toronto emotional speech set data/*AF_*/*.wav"):
        file_name = os.path.basename(file)
        emotion = file_name.split("_")[2][:-4]  # split and remove .wav
        sound.append(file)
        emo.append(emotion)

    return {"file": sound, "emotion": emo}


start_time = time.time()

Trial_dict = load_data()

print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))


X = pd.DataFrame(Trial_dict["file"])
y = pd.DataFrame(Trial_dict["emotion"])
X.shape, y.shape

y.value_counts()

# X_features = X[0].swifter.progress_bar(enable=True).apply(lambda x: extract_feature(x))

X_features =[]

for x in X[0]:
#     print(x)
    X_features.append(extract_feature(x))

X_features = pd.DataFrame(X_features)
#renaming the label column to emotion
y=y.rename(columns= {0: 'emotion'})
#concatinating the attributes and label into a single dataframe
data = pd.concat([X_features, y], axis =1)
data.head()

#reindexing to shuffle the data at random
data = data.reindex(np.random.permutation(data.index))
# Storing shuffled ravdess and tess data to avoid loading again
data.to_csv("TESS_FEATURES.csv")
starting_time = time.time()
data = pd.read_csv("./TESS_FEATURES.csv")
print("data loaded in " + str(time.time()-starting_time) + "ms")

print(data.head())

data.shape

#printing all columns
data.columns

#dropping the column Unnamed: 0 to removed shuffled index
data = data.drop('Unnamed: 0',axis=1)
data.columns

#separating features and target outputs
X = data.drop('emotion', axis = 1).values
y = data['emotion'].values
print(y)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X.shape, y.shape

np.unique(y)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
print("******************* svm ************************************************************************")
from sklearn.svm import SVC
svclassifier = SVC(kernel = 'rbf')

import time

starting_time = time.time()

svclassifier.fit(X_train, y_train)
print("Trained model in %s ms " % str(time.time() - starting_time))

train_acc = float(svclassifier.score(X_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svclassifier.score(X_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
         ('SVM', SVC(kernel='rbf'))]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Fit the pipeline to the training set: svc_scaled
svc_scaled = pipeline.fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(svc_scaled.score(X_test, y_test)))

cv_results2 = cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1)
print(cv_results2)
print("Average:", np.average(cv_results2))

train_acc = float(svc_scaled.score(X_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svc_scaled.score(X_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)

scaled_predictions = svc_scaled.predict(X_test)

print(classification_report(y_test,scaled_predictions))

acc = float(accuracy_score(y_test,scaled_predictions))*100
print("----accuracy score %s ----" % acc)

cm = confusion_matrix(y_test,scaled_predictions)

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix SVM');
ax.xaxis.set_ticklabels(tess_emotions);
ax.yaxis.set_ticklabels(tess_emotions);
plt.show()
print("******************* mlp   *************************************************************************")
from sklearn.neural_network import MLPClassifier

steps3 = [('scaler', StandardScaler()),
          ('MLP', MLPClassifier())]

pipeline_mlp = Pipeline(steps3)

mlp = pipeline_mlp.fit(X_train, y_train)

print('Accuracy with Scaling: {}'.format(mlp.score(X_test, y_test)))

mlp_train_acc = float(mlp.score(X_train, y_train)*100)
print("----train accuracy score %s ----" % mlp_train_acc)

mlp_test_acc = float(mlp.score(X_test, y_test)*100)
print("----test accuracy score %s ----" % mlp_train_acc)

mlp_res = cross_val_score(mlp, X, y, cv=cv, n_jobs=-1)
print(mlp_res)
print("Average:", np.average(mlp_res))

mlp_pred = mlp.predict(X_test)
print(mlp_pred)

print(classification_report(y_test,mlp_pred))

acc_mlp = float(accuracy_score(y_test,mlp_pred))*100
print("----accuracy score %s ----" % acc_mlp)

cm_mlp = confusion_matrix(y_test,mlp_pred)

ax= plt.subplot()
sns.heatmap(cm_mlp, annot=True, fmt='g', ax=ax);

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix (Multi Layer Perceptron)');
ax.xaxis.set_ticklabels(tess_emotions);
ax.yaxis.set_ticklabels(tess_emotions);
plt.show()
print("******************* KNN ********************************************************************")
from sklearn.neighbors import KNeighborsClassifier

steps4 = [('scaler', StandardScaler()),
          ('KNN', KNeighborsClassifier())]

pipeline_knn = Pipeline(steps4)

knn = pipeline_mlp.fit(X_train, y_train)

print('Accuracy with Scaling: {}'.format(knn.score(X_test, y_test)))

knn_train_acc = float(knn.score(X_train, y_train)*100)
print("----train accuracy score %s ----" % knn_train_acc)

knn_test_acc = float(knn.score(X_test, y_test)*100)
print("----test accuracy score %s ----" % knn_train_acc)

knn_res = cross_val_score(knn, X, y, cv=cv, n_jobs=-1)
print(knn_res)
print("Average:", np.average(knn_res))

knn_pred = knn.predict(X_test)
print(knn_pred)

print(classification_report(y_test,knn_pred))

acc_knn = float(accuracy_score(y_test,knn_pred))*100
print("----accuracy score %s ----" % acc_knn)

cm_knn = confusion_matrix(y_test,knn_pred)

ax= plt.subplot()
sns.heatmap(cm_knn, annot=True, fmt='g', ax=ax);

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix (K Nearest Neighbour)');
ax.xaxis.set_ticklabels(tess_emotions);
ax.yaxis.set_ticklabels(tess_emotions);
plt.show()
print("*******************Random forest *********************************************************************")
from sklearn.ensemble import RandomForestClassifier

rfm = RandomForestClassifier()
rfm_score = cross_val_score(rfm, X, y, cv=cv, n_jobs=-1)
print(rfm_score)
print("Average:", np.average(rfm_score))

rfm_res = rfm.fit(X_train, y_train)

rfm_train_acc = float(rfm_res.score(X_train, y_train)*100)
print("----train accuracy score %s ----" % rfm_train_acc)

rfm_test_acc = float(rfm_res.score(X_test, y_test)*100)
print("----test accuracy score %s ----" % rfm_test_acc)

rfm_pred = rfm_res.predict(X_test)
print(classification_report(y_test,rfm_pred))

acc = float(accuracy_score(y_test,rfm_pred))*100
print("----accuracy score %s ----" % acc)

cm_rfm = confusion_matrix(y_test,rfm_pred)

ax= plt.subplot()
sns.heatmap(cm_rfm, annot=True, fmt='g', ax=ax);

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix (Random Forest)');
ax.xaxis.set_ticklabels(tess_emotions);
ax.yaxis.set_ticklabels(tess_emotions);
plt.show()
print("******************* Gaussian NB ***********************************************************************")
from sklearn.naive_bayes import GaussianNB

nbm = GaussianNB().fit(X_train, y_train)

nbm_train_acc = float(nbm.score(X_train, y_train)*100)
print("----train accuracy score %s ----" % nbm_train_acc)

nbm_test_acc = float(nbm.score(X_test, y_test)*100)
print("----test accuracy score %s ----" % nbm_train_acc)

nbm_score = cross_val_score(nbm, X, y, cv=cv, n_jobs=-1)
print(nbm_score)
print("Average:", np.average(nbm_score))

nbm_pred = nbm.predict(X_test)
print(nbm_pred)

print(classification_report(y_test,nbm_pred))

acc_nbm = float(accuracy_score(y_test,nbm_pred))*100
print("----accuracy score %s ----" % acc_nbm)

cm_nbm = confusion_matrix(y_test,nbm_pred)

ax= plt.subplot()
sns.heatmap(cm_nbm, annot=True, fmt='g', ax=ax);

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix (Gaussian Naive Bayes)');
ax.xaxis.set_ticklabels(tess_emotions);
ax.yaxis.set_ticklabels(tess_emotions);
plt.show()

pickle.dump(mlp, open('mlp.pkl', 'wb'))







