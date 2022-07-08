#!/usr/bin/env python

import os
from glob import glob
import librosa
get_ipython().system('pip install librosa')
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.markers as mmarkers
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from word2number import w2n
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import torch
import numpy as np
from torch import nn
import sys
from torch import nn
from sklearn.model_selection import StratifiedShuffleSplit
import subprocess
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.data import Dataset
import copy
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from pomegranate import *
from tabulate import tabulate
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#################################################################
############################### STEP 2 ##########################
#################################################################

def read_wav(f):
    wav, _ = librosa.core.load(f, sr=None)
    return wav
    
def data_parser(directory):
    
    # Parse relevant dataset info
    files = glob(os.path.join(directory, "*.wav"))
    fnames = [f.split("/")[1].split("\\")[1].split(".wav")[0] for f in files]

    digits = []
    speakers = []
    
    for name in fnames:
        num = ''
        speak = ''
        for c in name:
            if c.isdigit():
                speak +=c
            else:
                num+=c
        
        speakers.append(speak)
        digits.append(num)
    
    wavs = [read_wav(f) for f in files]         
    return wavs, speakers, digits

L1, L2, L3 = data_parser("pr_lab2_2020-21_data/digits")


#################################################################
############################### STEP 3 ##########################
#################################################################


def extract_features(wavs, n_mfcc=13, Fs=16000):
    window = 25 * Fs // 1000
    step = 10*Fs//1000
    frames = [
        librosa.feature.mfcc(
            wav, Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
        ).T

        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    return frames


wavs, speakers, digits = data_parser("pr_lab2_2020-21_data/digits")
coeffs = extract_features(wavs)



mfcc_delta = [ librosa.feature.delta(coeff)
              for coeff in coeffs]

mfcc_delta2 = [ librosa.feature.delta(coeff, order=2)
              for coeff in coeffs]


#################################################################
############################### STEP 4 ##########################
#################################################################


# n1 --> 2
# n2 --> 6

ones = np.where(np.array(digits)=="one")[0]
twos = np.where(np.array(digits)=="two")[0]
threes = np.where(np.array(digits)=="three")[0]
fours = np.where(np.array(digits)=="four")[0]
fives = np.where(np.array(digits)=="five")[0]
sixs = np.where(np.array(digits)=="six")[0]
sevens = np.where(np.array(digits)=="seven")[0]
eights = np.where(np.array(digits)=="eight")[0]
nines = np.where(np.array(digits)=="nine")[0]


############################ first MFCC for n1 ##########################


nbins = 60;
coefficientToAnalyze = 4;
for j in range(3):

    f, (ax0, ax1, ax2, ax3,ax4) = plt.subplots(1, 5, figsize=(35, 10))
    
    ax0.hist(list(coeffs[twos[j*5]][0]), color ='lightskyblue', alpha = 0.7)
    ax0.set_title("Digit 2 of speaker "+str(speakers[twos[j*5]])+" first MFCC")
    ax1.hist(list(coeffs[twos[5*j+1]][0]), color ='lightskyblue', alpha = 0.7)
    ax1.set_title("Digit 2 of speaker "+str(speakers[twos[5*j+1]])+" first MFCC")
    ax2.hist(list(coeffs[twos[5*j+2]][0]), color ='lightskyblue', alpha = 0.7)
    ax2.set_title("Digit 2 of speaker "+str(speakers[twos[5*j+2]])+" first MFCC")
    ax3.hist(list(coeffs[twos[5*j+3]][0]), color ='lightskyblue', alpha = 0.7)
    ax3.set_title("Digit 2 of speaker "+str(speakers[twos[5*j+3]])+" first MFCC")
    ax4.hist(list(coeffs[twos[5*j+4]][0]), color ='lightskyblue', alpha = 0.7)
    ax4.set_title("Digit 2 of speaker "+str(speakers[twos[5*j+4]])+" first MFCC")



A11 = []
for i in range(len(twos)):
    A11.append(coeffs[twos[j*5]][0])
print("Standard Deviation of first MFCC for n1 is: ",np.std(np.array(A11)) )


######################## second MFCC for n1 ##################################


for j in range(3):

    f, (ax0, ax1, ax2, ax3,ax4) = plt.subplots(1, 5, figsize=(35, 10))
    
    ax0.hist(list(coeffs[twos[j*5]][1]), bins=60)
    ax0.set_title("Digit 2 of speaker "+str(speakers[twos[j*5]])+" second MFCC")
    ax1.hist(list(coeffs[twos[5*j+1]][1]), bins=60)
    ax1.set_title("Digit 2 of speaker "+str(speakers[twos[5*j+1]])+" second MFCC")
    ax2.hist(list(coeffs[twos[5*j+2]][1]), bins=60)
    ax2.set_title("Digit 2 of speaker "+str(speakers[twos[5*j+2]])+" second MFCC")
    ax3.hist(list(coeffs[twos[5*j+3]][1]), bins=60)
    ax3.set_title("Digit 2 of speaker "+str(speakers[twos[5*j+3]])+" second MFCC")
    ax4.hist(list(coeffs[twos[5*j+4]][1]), bins=60)
    ax4.set_title("Digit 2 of speaker "+str(speakers[twos[5*j+4]])+" second MFCC")



A12 = []
for i in range(len(twos)):
    A12.append(coeffs[twos[j*5]][1])
print("Standard Deviation of second MFCC for n1 is: ",np.std(np.array(A12)) )


###################### first MFCC for n2 ######################################


for j in range(3):
    
    f, (ax0, ax1, ax2, ax3,ax4) = plt.subplots(1, 5, figsize=(35, 10))
    
    ax0.hist(list(coeffs[sixs[j*5]][0]), bins=60)
    ax0.set_title("Digit 6 of speaker "+str(speakers[sixs[j*5]])+" first MFCC")
    ax1.hist(list(coeffs[sixs[5*j+1]][0]), bins=60)
    ax1.set_title("Digit 6 of speaker "+str(speakers[sixs[5*j+1]])+" first MFCC")
    ax2.hist(list(coeffs[sixs[5*j+2]][0]), bins=60)
    ax2.set_title("Digit 6 of speaker "+str(speakers[sixs[5*j+2]])+" first MFCC")
    ax3.hist(list(coeffs[sixs[5*j+3]][0]), bins=60)
    ax3.set_title("Digit 6 of speaker "+str(speakers[sixs[5*j+3]])+" first MFCC")
    if j<2:
        ax4.hist(list(coeffs[sixs[5*j+4]][0]), bins=60)
        ax4.set_title("Digit 6 of speaker "+str(speakers[sixs[5*j+4]])+" first MFCC")




A21 = []
for i in range(len(sixs)):
    A21.append(coeffs[sixs[j*5]][0])
print("Standard Deviation of first MFCC for n2 is: ",np.std(np.array(A21)) )


###################### second MFCC for n2 ######################################


for j in range(3):
    
    f, (ax0, ax1, ax2, ax3,ax4) = plt.subplots(1, 5, figsize=(35, 10))
    
    ax0.hist(list(coeffs[sixs[j*5]][1]), bins=60)
    ax0.set_title("Digit 6 of speaker "+str(speakers[sixs[j*5]])+" second MFCC")
    ax1.hist(list(coeffs[sixs[5*j+1]][1]), bins=60)
    ax1.set_title("Digit 6 of speaker "+str(speakers[sixs[5*j+1]])+" second MFCC")
    ax2.hist(list(coeffs[sixs[5*j+2]][1]), bins=60)
    ax2.set_title("Digit 6 of speaker "+str(speakers[sixs[5*j+2]])+" second MFCC")
    ax3.hist(list(coeffs[sixs[5*j+3]][1]), bins=60)
    ax3.set_title("Digit 6 of speaker "+str(speakers[sixs[5*j+3]])+" second MFCC")
    if j<2:
        ax4.hist(list(coeffs[sixs[5*j+4]][1]), bins=60)
        ax4.set_title("Digit 6 of speaker "+str(speakers[sixs[5*j+4]])+" second MFCC")



A22 = []
for i in range(len(sixs)):
    A22.append(coeffs[sixs[j*5]][1])
print("Standard Deviation of second MFCC for n2 is: ",np.std(np.array(A22)))


###### Εξάγετε για 2 εκφωνήσεις των n1 και n2 από 2 διαφορετικούς ομιλητές τα Mel Filterbank Spectral Coefficients (MFSCs). 
###### Αναπαραστήστε γραφικά τη συσχέτιση των MFSCs για την κάθε εκφώνηση. Συνεπώς, αποδίδουμε τον correlation coefficients πίνακα, για κάθε περίπτωση.

window = 25 * 16000 // 1000
step = 10*16000//1000
    
n1mfsc1 = librosa.feature.melspectrogram(y=wavs[twos[0]], n_mels=13, n_fft=round(0.025 * 16000),hop_length=window - step).T
n1mfsc2 = librosa.feature.melspectrogram(y=wavs[twos[5]], n_mels=13, n_fft=round(0.025 * 16000), hop_length=window - step).T
n2mfsc1 = librosa.feature.melspectrogram(y=wavs[sixs[0]], n_mels=13, n_fft=round(0.025 * 16000),hop_length=window - step).T
n2mfsc2 = librosa.feature.melspectrogram(y=wavs[sixs[5]], n_mels=13, n_fft=round(0.025 * 16000), hop_length=window - step).T



import seaborn as sns

mfsc = [n1mfsc1,n1mfsc2,n2mfsc1,n2mfsc2]
for j in range(2):
    
    f, (ax0, ax1) = plt.subplots(1, 2, figsize=(35, 10))
    
    sns.heatmap (np.corrcoef (mfsc[j], rowvar = False), ax = ax0)
    ax0.set_title("MFSC of Digit 2 of speaker "+str(speakers[twos[j*5]]))
    ax1 = sns.heatmap (np.corrcoef(coeffs[twos[j*5]], rowvar = False))
    ax1.set_title("MFCC of Digit 2 of speaker "+str(speakers[twos[5*j]]))
    plt.show ()

for j in range(2,4):
    k=0
    f, (ax0, ax1) = plt.subplots(1, 2, figsize=(35, 10))
    
    sns.heatmap (np.corrcoef (mfsc[j], rowvar = False), ax = ax0)
    ax0.set_title("MFSC of Digit 6 of speaker "+str(speakers[sixs[k*5]]))
    ax1 = sns.heatmap (np.corrcoef(coeffs[sixs[k*5]], rowvar = False))
    ax1.set_title("MFCC of Digit 6 of speaker "+str(speakers[sixs[5*k]]))
    k+=1
    plt.show ()    


#################################################################
############################### STEP 5 ##########################
#################################################################



def mean_std_vector(mfccs, deltas, ddeltas):
    
    vector = np.zeros((133,78)) #133:total number of ekfwnhseis
                                  
    i=0        
    for mfcc, delta, ddelta in zip(mfccs, deltas, ddeltas):
        mean_mfcc = np.mean(np.array(mfcc), axis=0)
        std_mfcc = np.std(np.array(mfcc), axis=0)
        mean_delta = np.mean(np.array(delta), axis=0)
        std_delta = np.std(np.array(delta), axis=0)
        mean_ddelta = np.mean(np.array(ddelta), axis=0)
        std_ddelta = np.std(np.array(ddelta), axis=0)
    
        #vector[i]= np.vstack(( mean_mfcc, std_mfcc,mean_delta, std_delta,mean_ddelta,std_ddelta))
        vector[i]= np.concatenate(( mean_mfcc, mean_delta,mean_ddelta,std_mfcc, std_delta,std_ddelta))
        i+=1
    return vector


vector = mean_std_vector(coeffs, mfcc_delta, mfcc_delta2)
print(vector.shape)


def myscatter(x,y,z,ax=None, m=None, **kw):
    if not ax: ax=plt.gca()
    if z == []:
        sc = ax.scatter(x,y,**kw)
    else:
        sc = ax.scatter(x,y,z,**kw)
    if (m is not None) and ((len(m)==len(x) or len(m)==len(x)+1)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                            marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


color_mapping = [ 'blue','red', 'green','gold','yellow','black','teal','orange','purple']
L = Line2D.markers
mark = []
markers = ['+','o','*','.','x','s','d','^','v','<','>','p','1','h','1']

def plot_my_scatter(vector, color_mapping, markers):
    fig = plt.figure(figsize = (10, 8))
    for i in range(9):

        x = []
        y = []
        z = []
        
        for j in nums[i]:
            x.append(vector[j,0])
            y.append(vector[j,1])

        scatter = myscatter(x, y,[], c=color_mapping[i], m=markers)

    plt.show()

plot_my_scatter(vector, color_mapping, markers)


#################################################################
############################### STEP 6 ##########################
#################################################################

print("The initial shape of our vector is: ", vector.shape)
pca = PCA(n_components=2)
vector2d = pca.fit_transform(vector)
print("The shape of the vector after PCA is: ", vector2d.shape)
print("The percentage of the initial variance for the two components is respectively:",  100*pca.explained_variance_ratio_[0],"%", 100*pca.explained_variance_ratio_[1],"%")  
print("The principal components store the percentage ", (100*pca.explained_variance_ratio_[0])+(100*pca.explained_variance_ratio_[1]),"% of the initial information")


####################### scatterplot for 2 dimensions ############



plot_my_scatter(vector2d, color_mapping, markers)


#################### 3 dimensions #################################




print("The initial shape of our vector is: ", vector.shape)
pca = PCA(n_components=3)
vector3d= pca.fit_transform(vector)
print("The shape of the vector after PCA is: ", vector3d.shape)
print("The percentage of the initial variance for the 3 components is respectively:",  100*pca.explained_variance_ratio_[0],"%", 100*pca.explained_variance_ratio_[1],"%", 100*pca.explained_variance_ratio_[2],"%")  
print("The principal components store the percentage ", (100*pca.explained_variance_ratio_[0])+(100*pca.explained_variance_ratio_[1])+(100*pca.explained_variance_ratio_[2]),"% of the initial information")




def plot_my_scatter_3d(vector2d, color_mapping, markers):
    
    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(1,1,1,projection = '3d') 
    ax.set_xlabel("Componenct 1")
    ax.set_ylabel("Componenct 2")
    ax.set_zlabel("Componenct 3")
    ax.set_title("3d PCA")
    for i in range(9):

        x = []
        y = []
        z = []
        
        for j in nums[i]:
            x.append(vector[j,0])
            y.append(vector[j,1])
            z.append(vector[j,2])
            
        scatter = myscatter(x, y, z, c=color_mapping[i], m=markers)
    plt.show()
    
plot_my_scatter_3d(vector3d, color_mapping, markers)


#################################################################
############################### STEP 7 ##########################
#################################################################

# we consider that the dataset is the 133 speeches of 78 features
#with this function we compute the mean and the var values 
#of the 87 features of each digit

def features_mean_var(vector,y):
    
    #print("initially vector is: ", vector.shape)
    X_mean = [np.zeros(78)]*9
    X_var = [np.zeros(78)]*9
    
    ones = np.where(np.array(y)==1)[0]
    twos = np.where(np.array(y)==2)[0]
    threes = np.where(np.array(y)==3)[0]
    fours = np.where(np.array(y)==4)[0]
    fives = np.where(np.array(y)==5)[0]
    sixs = np.where(np.array(y)==6)[0]
    sevens = np.where(np.array(y)==7)[0]
    eights = np.where(np.array(y)==8)[0]
    nines = np.where(np.array(y)==9)[0]
    
    nums = [ones, twos, threes, fours, fives, sixs, sevens, eights, nines]
    
    c = 0
    for i in nums:
        sum = 0
        for j in i:
            temp = vector[j]
            #print("the length of temp is:", len(temp), temp)
            sum+=temp
        X_mean[c] = sum/len(nums[c])
        c+=1
    
    var_values=[[]]*9
    for i in range(9):
        var = [0]*78
        all_features = [[]]*len(nums[i])
        for j in range(len(nums[i])):
                all_features[j] = vector[nums[i][j]]
        for k in range(78):
            var[k] = np.var(np.array(all_features)[:,k])
        var_values[i] = var
    X_var= var_values
        
    return X_mean, X_var


int_digits = np.zeros(len(digits),dtype=np.int8)
for i in range(len(digits)):
    int_digits[i] = int(w2n.word_to_num(digits[i]))

X_train, X_test, y_train, y_test = train_test_split(vector, int_digits, test_size=0.3,random_state=42)

#############################################################################################
######################### Our Naive Bayes Classifier ########################################
#############################################################################################

def calculate_priors(y):
    prior_probabilities = np.zeros(9)
    for c in y:
        prior_probabilities[c-1] += 1 
    prior_probabilities = prior_probabilities/len(y)
    return prior_probabilities 


class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance


    def fit(self, X, y):
        
        self.X_mean_, self.X_var_ = features_mean_var(X,y)
        self.y = y
        
        return self.X_mean_, self.X_var_


    def predict(self, X):
        
        MEAN = np.array(self.X_mean_)
        VAR = np.array(self.X_var_)
        priors = calculate_priors(self.y)
        predictions = np.zeros(len(X))
        
        for i in range(len(X)):
            conditional_probs = np.zeros(39)
            product_probs = np.zeros(9)
            class_probabilities = np.zeros(9)

            for j in range(9):
                VAR[j][VAR[j] == 0] = 1
                
                A = -(X[i]-MEAN[j])**2
                B = 2*(VAR[j])
                C = np.sqrt(2*math.pi*VAR[j])
                D = A/B
                conditional_probs = (np.exp(D))/C
                
                product_probs = np.prod(conditional_probs, where = conditional_probs != 0)
                class_probabilities[j] = product_probs*priors[j]

            prediction = np.argmax(class_probabilities)
            predictions[i] = prediction+1
        return predictions
    
    def score(self, X, y):
        s = 0
        predictions = self.predict(X)
        for j in range(len(y)):
            if predictions[j] == y[j]:
                s+=1
        return s/(len(y))


X_train = preprocessing.scale(X_train)
scaler = preprocessing.StandardScaler().fit(X_train)
X_test = scaler.transform(X_test)

X_data = np.concatenate((X_train, X_test), axis = 0)
y_data = np.concatenate((y_train,y_test), axis = 0)

#5-cross-validation for our classifier
def return_mean_scores(Classifier,X,y):
    scores = cross_val_score(Classifier, X, y, 
                             cv=KFold(n_splits=5), 
                             scoring="accuracy")
    return np.mean(np.array(scores))

return_mean_scores(CustomNBClassifier(),X_train, y_train)


#accuracy in test set for our classifier
my_NBC = CustomNBClassifier()
my_NBC.fit(X_train, y_train)
my_NBC.score(X_test, y_test)

##########################################
#5-cross-validation for scikit GaussianNB
##########################################
classifier = GaussianNB()
scores = cross_val_score(classifier, X_train, y_train, 
                         cv=KFold(n_splits=5), 
                         scoring="accuracy")
print(np.mean(np.array(scores)))

#accuracy in test set for scikit GaussianNB
gaussian = GaussianNB( var_smoothing=0.1)
gaussian.fit(X_train, y_train)
gaussian.score(X_test, y_test)

####################################################
##################### SVM ##########################
####################################################

SVM = SVC(C= 980, gamma= 'scale', kernel= 'rbf')
#5-fold-cross-validation
return_mean_scores(SVM,X_train, y_train)
#accuracy in test set
SVM.fit(X_train, y_train)
SVM.score(X_test, y_test)

######################################################
######################### KNN ########################
######################################################

KNN = KNeighborsClassifier(n_neighbors=1, weights='uniform',algorithm='ball_tree')
#5-fold-cross-validation
return_mean_scores(KNN,X_train, y_train)
#accuracy in test set
KNN.fit(X_train, y_train)
KNN.score(X_test, y_test)

######################################################
######################### MLP ########################
######################################################

MLP = MLPClassifier(activation= 'tanh', alpha=1e-05, hidden_layer_sizes=(120,), learning_rate= 'constant', max_iter=1100, solver='adam')
#5-fold-cross-validation
return_mean_scores(MLP,X_train, y_train)
#accuracy in test set
MLP.fit(X_train, y_train)
MLP.score(X_test, y_test)

#################################################################
############################### STEP 8 ##########################
#################################################################


######## input sin target cos ###################################

time_points = torch.linspace(0, 1000 * np.pi, 4500)
sin_points = torch.sin(time_points)
sin_points = sin_points.view(1,450,10)
x = sin_points

y = torch.cos(torch.linspace(0, 1000 * np.pi, 4500))
y = y.view(1,450,10)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]



input_size = 10
hidden_layer_size = 32
output_size = 10
lr = .001
epochs = 50

# optimizer and loss function
model = LSTM(input_size,hidden_layer_size,output_size)
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# training 
for i in range(epochs):
    for j in range(450):
        seq = x[:,j,:]
        labels = y[:,j,:]

        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)
        optimizer.zero_grad()

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()


tempx = torch.sin(torch.linspace(0, 2*np.pi, 10))
model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                torch.zeros(1, 1, model.hidden_layer_size))
pred = model(tempx.view(1,1,10))

lin02 = np.linspace(0,2*np.pi,10)
plt.plot(lin02,pred.data.numpy().flatten(),'royalblue',label = 'prediction')
plt.plot(lin02,np.sin(lin02),'coral', label = 'sin')
plt.legend(loc = 'best')
plt.show()



time_points = torch.linspace(0, 1000 * np.pi, 4500)
sin_points = torch.sin(time_points)
sin_points = sin_points.view(1,450,10)
x = sin_points

y = torch.cos(torch.linspace(0, 1000 * np.pi, 4500))
y = y.view(1,450,10)


#################################################################
############################### STEP 9 ##########################
#################################################################

def parse_free_digits(directory):
    # Parse relevant dataset info
    files = glob(os.path.join(directory, "*.wav"))
    #print(files)
    #free-spoken-digit-dataset-master/recordings\\0_george_0.wav
    fnames = [f.split("\\")[1].split(".")[0].split("_") for f in files]
    ids = [f[2] for f in fnames]
    y = [int(f[0]) for f in fnames]
    speakers = [f[1] for f in fnames]
    _, Fs = librosa.core.load(files[0], sr=None)

    def read_wav(f):
        wav, _ = librosa.core.load(f, sr=None)

        return wav

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, Fs, ids, y, speakers




def extract_features(wavs, n_mfcc=6, Fs=8000):
    # Extract MFCCs for all wavs
    window = 30 * Fs // 1000
    step = window // 2
    frames = [
        librosa.feature.mfcc(
            wav, Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
        ).T

        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    return frames


# In[16]:


def split_free_digits(frames, ids, speakers, labels):
    print("Splitting in train test split using the default dataset split")
    # Split to train-test
    X_train, y_train, spk_train = [], [], []
    X_test, y_test, spk_test = [], [], []
    test_indices = ["0", "1", "2", "3", "4"]

    for idx, frame, label, spk in zip(ids, frames, labels, speakers):
        if str(idx) in test_indices:
            X_test.append(frame)
            y_test.append(label)
            spk_test.append(spk)
        else:
            X_train.append(frame)
            y_train.append(label)
            spk_train.append(spk)

    return X_train, X_test, y_train, y_test, spk_train, spk_test




def make_scale_fn(X_train):
    # Standardize on train data
    scaler = StandardScaler()
    scaler.fit(np.concatenate(X_train))
    print("Normalization will be performed using mean: {}".format(scaler.mean_))
    print("Normalization will be performed using std: {}".format(scaler.scale_))
    def scale(X):
        scaled = []

        for frames in X:
            scaled.append(scaler.transform(frames))
        return scaled
    return scale



def parser(directory, n_mfcc=6):
    wavs, Fs, ids, y, speakers = parse_free_digits(directory)
    frames = extract_features(wavs, n_mfcc=n_mfcc, Fs=Fs)
    X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(
        frames, ids, speakers, y
    )

    return X_train, X_test, y_train, y_test, spk_train, spk_test


X_train, X_test, y_train, y_test, spk_train, spk_test = parser('free-spoken-digit-dataset-master/recordings')

X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, stratify = y_train, test_size = .2) #split train from validation 80%-20%
print("If using all data to calculate normalization statistics")
scale_fn = make_scale_fn(X_train + X_dev + X_test)
print("If using X_train + X_dev to calculate normalization statistics")
scale_fn = make_scale_fn(X_train + X_dev)
print("If using X_train to calculate normalization statistics")
scale_fn = make_scale_fn(X_train)
X_train = scale_fn(X_train)
X_dev = scale_fn(X_dev)
X_test = scale_fn(X_test)



#test if the stratified split is ok
#each set contains approximately the same percentage of samples of each 
#target class as the complete set.
y = np.array(y_train)
L = []
for i in range(0,10):
    L.append((np.where(y == i)[0]).shape[0])
print(L)

y = np.array(y_test)
L = []
for i in range(0,10):
    L.append((np.where(y == i)[0]).shape[0])
print(L)

y = np.array(y_dev)
L = []
for i in range(0,10):
    L.append((np.where(y == i)[0]).shape[0])
print(L)

#################################################################
############################### STEP 10 #########################
#################################################################

def hmm(X, n_states, n_mixtures, gmm = True):
    X_stacked = np.vstack(X)

    dists = [] # list of probability distributions for the HMM states
    for i in range(n_states):
        if gmm:
            a = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_mixtures, np.float_(X_stacked))
        else:
            a = MultivariateGaussianDistribution.from_samples(np.float_(X_stacked))
        dists.append(a)

    # trans_mat = [] # your transition matrix
    trans_mat = np.reshape([0.5 if i==j or j==i+1 else 0 for i in range(n_states) for j in range(n_states)],(n_states,n_states)) 

    starts = np.zeros(n_states)# your starting probability matrix
    ends = np.zeros(n_states) # your ending probability matrix
    #init
    starts[0] = 1
    ends[-1] = 1


    # Define the GMM-HMM
    model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends, state_names=['s{}'.format(i) for i in range(n_states)])

    # Fit the model
    model.fit(X, max_iterations=5)

    # Predict a sequence
    # sample = [] # a sample sequence
    # logp, _ = model.viterbi(sample) # Run viterbi algorithm and return log-probability
    
    return model

#################################################################
############################### STEP 11 #########################
#################################################################

X_train_per_digit = []
for digit in range(10):
    X_tr_i = np.take(X_train, [i for i in range(len(X_train)) if y_train[i] == digit], axis=0)
    # X_tr_i = np.take(X_train, np.where(np.array(y_train) == digit), axis=0)
    X_train_per_digit.append(X_tr_i)


def train(X, n_states,n_mixtures, gmm=True):
    models = []
    for i in X:
        models.append(hmm(i, n_states, n_mixtures))
    return models

#################################################################
############################### STEP 12 #########################
#################################################################

def eval_models(models, X_val, y_val, n):
    conf_matrix = np.zeros((10, 10)) # confusion matrix
    y_preds = np.zeros(n, dtype='int') # predictions
    for i in range(n):
        logs = np.zeros(10)
        # Evaluate the sample in each model and decode it to the digit with the highest log-likelihood.
        for j in range(10):
            logp, _ = models[j].viterbi(X_val[i]) # Run viterbi algorithm and return log-probability
            logs[j] = logp
        y_preds[i] = np.argmax(logs)
        conf_matrix[y_val[i], y_preds[i]] += 1
    acc = sum(y_preds == y_val) / n
    
    return acc, conf_matrix
    
accs = joblib.load('accs.pkl')
# models = joblib.load('models.pkl')
print(accs)


# In[14]:


n_states = 4
n_mixtures = 2
# models = train(X_train_per_digit, n_states, n_mixtures)
models = joblib.load('models.pkl')
acc_dev, conf_matrix_dev = eval_models(models, X_dev, y_dev, len(X_dev))
acc_test, conf_matrix_test = eval_models(models, X_test, y_test, len(X_test))

#################################################################
############################### STEP 13 #########################
#################################################################

ax = sns.heatmap(conf_matrix_dev,annot=True)
ax = sns.heatmap(conf_matrix_test,annot=True)

#################################################################
############################### STEP 14 #########################
#################################################################


# we use our GPU for faster training 
device = torch.device("cuda")
   

#custom dataset class that has to implement three functions 
#__init__
#__getitem__
#__len__
class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
            feats: Python list of numpy arrays that contain the sequence features.
                   Each element of this list is a numpy array of shape seq_length x feature_dimension
            labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        self.lengths =  [feats[i].shape[0] for i in range(len(feats))]

        self.feats = self.zero_pad_and_stack(feats)
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype('int64')
        
        ####################### 14.8 Bonus Part ##################
        ####### we sort the sequences in deacrising length #######
        ############## we order the labels the right way #########
        self.lengths = np.array(self.lengths)
        sort_length_index = np.argsort(self.lengths)[::-1]
        self.feats = self.feats[sort_length_index][:][:]
        self.lengths = self.lengths[sort_length_index]
        self.labels = self.labels[sort_length_index]
        ###########################################################
    
    def zero_pad_and_stack(self, x):
        """
            This function performs zero padding on a list of features and forms them into a numpy 3D array
            returns
                padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        padded = np.zeros((len(x), max(self.lengths), x[0].shape[1] ))
        # --------------- Insert your code here ---------------- #
        for i in range(len(x)):
            for j in range(len(x[i])):
                for k in range(len(x[i][j])):
                    padded[i][j][k] = x[i][j][k] 
        return padded

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)


class BasicLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, bidirectional=False, dropout = 0):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size

        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers
        self.input_dim = input_dim
        self.rnn_size = rnn_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=rnn_size,
                          num_layers=num_layers, batch_first=True, 
                            bidirectional = bidirectional, dropout = dropout) #lstm
        # we want N x L x D form (where N batch index) so batch_first=True
        self.fc_1 =  nn.Linear(self.feature_size, 3) #fully connected 1
        #self.fc = nn.Linear(128, num_classes) #fully connected last layer
        self.fc = nn.Linear(3, self.output_dim)
        self.relu = nn.ReLU()
        
        
        
        
    def forward(self, x, lengths):
        """ 
            x : 3D numpy array of dimension N x L x D
                N: batch index 
                L: sequence index
                D: feature index

            lengths: N x 1
         """
        #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        #x = np.array(x)
        # --------------- Insert your code here ---------------- #
        
        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network
        self.lengths = lengths
        
        h_0 = Variable(torch.zeros(self.num_layers, len(x), self.feature_size)).float().to(device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, len(x), self.feature_size)).float().to(device) #internal state
        
        
        ############################ Bonus part 14.8 ########################
        x = pack_padded_sequence(x, lengths.cpu(), batch_first = True)
        #####################################################################
        
        # Propagate input through LSTM
        if self.bidirectional:
            h_0 = h_0.reshape((int(2*h_0.shape[0]), int(h_0.shape[1]), int(h_0.shape[2]/2)))
            c_0 = c_0.reshape((int(2*c_0.shape[0]), int(c_0.shape[1]), int(c_0.shape[2]/2)))
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        #hn = hn.view(-1, self.rnn_size) #reshaping the data for Dense layer next
        
        ############################ Bonus part 14.8 ########################
        output, _ = pad_packed_sequence(output, batch_first=True)
        #####################################################################

        out = self.last_timestep(output,self.lengths, self.bidirectional)
        out = self.relu(out)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        last_outputs = self.fc(out) #Final Output
        
        
        return last_outputs

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()


# In[120]:


import matplotlib.pyplot as plt 

def train_model(rnn_size,num_layers, epochs, batch_size, learning_rate, bidirectional = False, dropout = 0.0, regularization = 0.0, early_stopping = False, title = 'LSTM'):
      
    #dropout and regularization parameter is for ex 14.5
    #early_stopping parameter is for ex 14.6
    #bidirectional parameter is for ex 16.7
    
    #load our model with the function we created above
    model = BasicLSTM(input_dim = 6, rnn_size = rnn_size, output_dim = 10, num_layers = num_layers, bidirectional = bidirectional, dropout = dropout)
    #load model to GPU in order to save time
    model.to(device)
    

    #criterion:we try to find the parameters that maximize the probability of the training data
    criterion = nn.CrossEntropyLoss()
    #Optimization algorithm changes the attributes of the neural network such as weights  
    #to reduce the losses. Optimizers are used to solve optimization problems by minimizing the function.
    optimizer = torch.optim.SGD(model.parameters(), weight_decay = regularization, lr = learning_rate)
    training_loss = []
    validation_loss = []
    
    ########################## 14.6 ###################################
    ####### Add  Early Stopping and Checkpoints to the model ##########
    ###################################################################
    
    #In order to avoid overtraining in neural networks we use early stopping
    #we stop the training process at the point when performance on a validation dataset starts to degrade
    #we set a big pseudovalue to early variable and we compare it with the validation loss. If the loss
    #in validation set increases from one epoch to another we activate the early stopping and we stop training
    early = 10000
    stopped_break = 0 
    stop_training = False

    #checking parameters
    best_validation_model = None
    best_validation_score = 10

    # training, calclulating loss for train / val sets
    for epoch in range(epochs):
        trloss = 0 
        valoss = 0
        
        #train set
        for i in range(int(len(X_train)/ batch_size)):
            X_batch = X_train[i * batch_size: (i + 1) * batch_size]
            y_batch = y_train[i * batch_size: (i + 1) * batch_size]
            
            ################ get the inputs #######################
            dataset = FrameLevelDataset(X_batch, y_batch)
            X_batch = dataset.zero_pad_and_stack(X_batch)
            X_batch = torch.tensor(X_batch).float().to(device)
            lengths = dataset.lengths
            lengths = torch.tensor(lengths).long().to(device)
            y_pred = model(X_batch,lengths).to(device)
            y_batch = torch.tensor(y_batch).long().to(device)
            
            ############# zero the parameter gradients ###########
            optimizer.zero_grad()
            ############# forward + backward + optimize ##########
            #https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            trloss +=  loss

        # append train loss
        ###### print every len(X_train)/ batch_size mini-batches
        training_loss.append(trloss.item()/(len(X_train)/ batch_size))
        print('Train loss for epoch ',epoch+1,":",trloss.item()/(len(X_train)/ batch_size))

        # validation set
        dataset = FrameLevelDataset(X_dev, y_dev)
        X_dev_batch = dataset.zero_pad_and_stack(X_dev)
        X_dev_batch = torch.tensor(X_dev_batch).float().to(device)
        lengths = dataset.lengths
        lengths = torch.tensor(lengths).long().to(device)
        y_dev_pred = model(X_dev_batch,lengths).to(device)
        y_dev_batch = torch.tensor(y_dev).long().to(device)
        
        # append validation loss
        valoss = criterion(y_dev_pred,y_dev_batch)
        validation_loss.append(valoss.item())
        print('Validation loss for epoch ',epoch+1,":",valoss.item())
        print()

        ########################## 14.6 #############################
        ########## this is our early stopping condition #############
        #### if early stopping is activated the we do the checkpoint#
        ############################################################# 
        if early_stopping == True and epoch > 50:
            if early > valoss.item():
                early = valoss.item()
            else:
                stopped_break = epoch
                stop_training = True
                break
        if stop_training == True:
            break

        ########## we save the model with the minimum loss #########
        ########## in validation set using torch.save ##############
        ############################################################
        if valoss.item() < best_validation_score:
            best_validation_score = valoss.item()
            best_validation_model = copy.deepcopy(model)
            torch.save(model,'best')

    epoch_linspace = np.linspace(1,epoch+1,epoch+1)
        
    ################### Loss/Epoch diagram for train and val dataset ##########
    plt.plot(epoch_linspace,training_loss,'royalblue',label = 'Training loss')
    plt.plot(epoch_linspace,validation_loss,'moccasin',label = 'Validation loss')
    plt.title(title)
    plt.legend(loc = 'best')
    plt.show()
    return model



####################### 14.3, 14.4 ####################################

rnn_size = 64
num_layers = 2
epochs = 100
batch_size = 27
learning_rate = 0.05
plain_model = train_model(rnn_size,num_layers,epochs,batch_size,learning_rate,regularization=0,dropout=0,early_stopping = False, bidirectional=False ,title = 'Plain LSTM')

######################### 14.5 ######################################

regularization = 0.0001 
dropout = 0.01
reg_drop_model = train_model(rnn_size,num_layers,epochs,batch_size,learning_rate,regularization=regularization,dropout=dropout,title = 'Regularization and Dropout')

############################## 14.6 #################################

early_reg_drop_model = train_model(rnn_size,num_layers,epochs,batch_size,learning_rate,regularization=regularization,dropout=dropout,early_stopping = True,title = 'Early Stopping Regularization and Dropout')

############################## 14.7 #################################

bidirectional = True
bidirectional_model = train_model(rnn_size,num_layers,epochs,batch_size,learning_rate,regularization=regularization,dropout=dropout,early_stopping = True, bidirectional=True ,title = 'Bidirectional LSTM with Early Stopping Regularization and Dropout')

############################## 14.8 #################################

bidirectional = True
optimized_bidirectional_model = train_model(rnn_size,num_layers,epochs,batch_size,learning_rate,regularization=regularization,dropout=dropout,early_stopping = True, bidirectional=True ,title = 'Pack padded Bidirectional LSTM with Early Stopping Regularization and Dropout')


# we chose as best model the optimized Bidirectional LSTM with Early Stopping Regularization and Dropout called "optimized_bidirectional_model"

model = optimized_bidirectional_model

dataset = FrameLevelDataset(X_dev, y_dev)
X_dev_batch = dataset.zero_pad_and_stack(X_dev)
X_dev_batch = torch.tensor(X_dev_batch).float().to(device)
lengths = dataset.lengths
lengths = torch.tensor(lengths).long().to(device)
y_dev_pred = model(X_dev_batch,lengths).to(device)
y_dev_batch = torch.tensor(y_dev).long().to(device)

y_pred = y_dev_pred.cpu().detach().numpy()
y_pred = np.array([np.argmax(y_pred[i]) for i in range(y_pred.shape[0])])
y_batch = y_dev_batch.cpu().detach().numpy()
print("Validation Set Accuracy ",round(accuracy_score(y_batch,y_pred),3))



cm = confusion_matrix(y_batch, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Validation Set Confusion Matrix")
plt.show()




dataset = FrameLevelDataset(X_test, y_test)
X_test_batch = dataset.zero_pad_and_stack(X_test)
X_test_batch = torch.tensor(X_test_batch).float().to(device)
lengths = dataset.lengths
lengths = torch.tensor(lengths).long().to(device)
y_test_pred = model(X_test_batch,lengths).to(device)
y_test_batch = torch.tensor(y_test).long().to(device)


y_pred = y_test_pred.cpu().detach().numpy()
y_pred = np.array([np.argmax(y_pred[i]) for i in range(y_pred.shape[0])])
y_batch = y_test_batch.cpu().detach().numpy()
print("Test set Accuracy",round(accuracy_score(y_batch,y_pred),3))


cm = confusion_matrix(y_batch, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Test Set Confusion Matrix")
plt.show()

