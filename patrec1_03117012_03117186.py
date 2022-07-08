#!/usr/bin/env python

##########################################################################################################
########################################### Andreas Vezakis 03117186 #####################################
########################################### Tsakanika Christina 03117012 #################################
##########################################################################################################

import numpy as np
import math
import matplotlib.pyplot
import random
from sklearn.base import BaseEstimator, ClassifierMixin
import nltk
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import  VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

##########################################################################################################
########################################### step1 ########################################################
##########################################################################################################


X_train = []
X_test = []
y_train = []
y_test = []

with open("data/train.txt", "r") as f1:
   
    # reading each line    
    for line in f1:
        
        y_train.append(int(line[0]))

        line = line.split(' ', 1)[1] #take off the label        
        
        features = [float(f) for f in line.split()] #features of each digit
        X_train.append(features)

X_train = np.array(X_train)
y_train = np.array(y_train)
f1.close()
print(y_train.shape, X_train.shape)


# In[110]:


with open("data/test.txt", "r") as f2:
   
    # reading each line    
    for line in f2:
        
        y_test.append(int(line[0]))

        line = line.split(' ', 1)[1] #take off the label        
        
        features = [float(f) for f in line.split()] #features of each digit
        X_test.append(features)

X_test = np.array(X_test)
y_test = np.array(y_test)
f2.close()
print(y_test.shape, X_test.shape)


##########################################################################################################
########################################### step2 ########################################################
##########################################################################################################

matplotlib.pyplot.imshow(X_train[131].reshape(16,16)) #we print with the proper form of 16x16 array


##########################################################################################################
########################################### step3 ########################################################
##########################################################################################################

#In "indexes" list of lists, the first list stores the indexes of the zero features, the second list the indexes
#of one features... and the last list the indexes of features 9
indexes = [np.where(y_train==0)[0],np.where(y_train==1)[0],np.where(y_train==2)[0],np.where(y_train==3)[0],
           np.where(y_train==4)[0],np.where(y_train==5)[0],np.where(y_train==6)[0],np.where(y_train==7)[0],
           np.where(y_train==8)[0],np.where(y_train==9)[0]]

#for each each digit we choose a random array of its features
random_samples = [random.choice(indexes[0]),random.choice(indexes[1]),random.choice(indexes[2]),
                 random.choice(indexes[3]),random.choice(indexes[4]),random.choice(indexes[5]),
                 random.choice(indexes[6]),random.choice(indexes[7]),random.choice(indexes[8]),
                 random.choice(indexes[9])]



#we chose to show the random digits in two lines and five columns form
for i in range(2):
    f, (ax0, ax1, ax2, ax3,ax4) = matplotlib.pyplot.subplots(1, 5, figsize=(20, 20))

    ax0.imshow(X_train[random_samples[i*5]].reshape(16,16))
    ax0.set_title(y_train[random_samples[i*5]], fontsize=14)
    ax1.imshow(X_train[random_samples[i*5+1]].reshape(16,16))
    ax1.set_title(y_train[random_samples[i*5+1]], fontsize=14)
    ax2.imshow(X_train[random_samples[i*5+2]].reshape(16,16))
    ax2.set_title(y_train[random_samples[i*5+2]], fontsize=14)
    ax3.imshow(X_train[random_samples[i*5+3]].reshape(16,16))
    ax3.set_title(y_train[random_samples[i*5+3]], fontsize=14)
    ax4.imshow(X_train[random_samples[i*5+4]].reshape(16,16))
    ax4.set_title(y_train[random_samples[i*5+4]], fontsize=14)

##########################################################################################################
########################################### step4 ########################################################
##########################################################################################################


sum = 0
for i in indexes[0]: #indexes[0] stores the indexes of all zero samples
    train_zero = X_train[i].reshape(16,16) #we reshape the sample in oreder to find the [10][10] pixel
    sum+=train_zero[10][10] # we sum all the values of pixels [10][10] of zero samples
mean_value = sum/len(indexes[0]) #we divide it by the total number of zero samples
print(mean_value)

##########################################################################################################
########################################### step5 ########################################################
##########################################################################################################


values = []
for i in indexes[0]:
    train_zero = X_train[i].reshape(16,16)
    values.append(train_zero[10][10])
print(np.var(values)) #we use the np.var in order to find the variance of the array of [10][10] zero pixels


##########################################################################################################
########################################### step6 ########################################################
##########################################################################################################

#we will add all the arrays with zero features (sum array) and then we will divide it with the number of 
#zero features in order to have the mean value of each pixel. The output will be mean_values array
sum = np.zeros((16,16))

for i in indexes[0]:
    train_zero = X_train[i].reshape(16,16)
    sum+=train_zero
mean_values = sum/len(indexes[0]) 
print(mean_values[10][10]) #test if output is equal with step4

zeros = [[]]*len(indexes[0]) #an array with all the features of zero label
zero_var = [0]*256 #the array where we will store the varriance of each pixel of zero features
for i in range(len(indexes[0])):
    zeros[i] = X_train[indexes[0][i]]
for i in range(len(zeros[0])):
    zero_var[i] = np.var(np.array(zeros)[:,i]) 
print(np.array(zero_var).reshape(16,16)[10][10]) #test if the output is equal with step5


##########################################################################################################
########################################### step7 ########################################################
##########################################################################################################


matplotlib.pyplot.imshow(mean_values.reshape(16,16))

##########################################################################################################
########################################### step8 ########################################################
##########################################################################################################

matplotlib.pyplot.imshow(np.array(zero_var).reshape(16,16))

##########################################################################################################
########################################### step9 ########################################################
##########################################################################################################

#(a)

mean_values = [np.zeros((16,16))]*10
c = 0
for j in indexes:
    sum = 0
    for i in j:
        train_zero = X_train[i].reshape(16,16)
        sum+=train_zero
    mean_values[c] = sum/len(indexes[c])
    c+=1
print(mean_values[0][10][10]) #test if the output for specific pixel is equal to the previous steps

var_values = [[]]*10
for i in range(10):
    var = [0]*256
    all_features = [[]]*len(indexes[i])
    for j in range(len(indexes[i])):
        all_features[j] = X_train[indexes[i][j]]
    for k in range(256):
        var[k] = np.var(np.array(all_features)[:,k])
    var_values[i] = var

print(np.array(var_values[0]).reshape(16,16)[10][10]) #test if the output for specific pixel is equal to the previous steps

#(b)
#we chose to print the mean values array of each digit in the form of two lines and five columns
for i in range(2):
    f, (ax0, ax1, ax2, ax3,ax4) = matplotlib.pyplot.subplots(1, 5, figsize=(20, 20))

    ax0.imshow(mean_values[i*5].reshape(16,16))
    ax0.set_title("mean "+str(i*5), fontsize=14)
    ax1.imshow(mean_values[i*5+1].reshape(16,16))
    ax1.set_title("mean "+str(i*5+1), fontsize=14)
    ax2.imshow(mean_values[i*5+2].reshape(16,16))
    ax2.set_title("mean "+str(i*5+2), fontsize=14)
    ax3.imshow(mean_values[i*5+3].reshape(16,16))
    ax3.set_title("mean "+str(i*5+3), fontsize=14)
    ax4.imshow(mean_values[i*5+4].reshape(16,16))
    ax4.set_title("mean "+str(i*5+4), fontsize=14)

##########################################################################################################
########################################### step10 #######################################################
##########################################################################################################

distances =[] #first element: distance from zero mean values, second:distance from one mean values etc...
test_sample = X_test[101].reshape(16,16)
for i in range(10): #we compute the distance of the 101st sample from the mean values of each digit
    dist = np.linalg.norm(mean_values[i] - test_sample)
    distances.append(dist)
category = distances.index(min(distances)) #we chose the digit with the minimum distance from the sample
print(category)
print(y_test[101])
matplotlib.pyplot.imshow(test_sample)

#finally the prediction (category) != y_test[101], so the classification was wrong.

##########################################################################################################
########################################### step11 #######################################################
##########################################################################################################

#(a) test set classifiacation
acc = 0
for j in range(len(y_test)):
    distances =[]
    test_sample = X_test[j].reshape(16,16)
    for i in range(10):
        dist = np.linalg.norm(mean_values[i] - test_sample) #compute the distance
        distances.append(dist)                              #of each sample from the mean values
    category = distances.index(min(distances)) #and chose the digit with the minimum
    if category == y_test[j]: #if prediction is right
        acc+=1

#(b)
print("accuracy is: ", acc/(len(y_test))) #accuracy: #right predictions/#samples
    
##########################################################################################################
########################################### step12 #######################################################
##########################################################################################################

#we fill the given functions from lib.py
def euclidean_distance(s, m):
    distance = np.linalg.norm(m - s)
    return distance

def euclidean_distance_classifier(X, X_mean):
    predictions = []
    for j in range(len(X)):
        distances =[]
        test_sample = X[j].reshape(16,16)
        for i in range(10):
            dist = euclidean_distance(X_mean[i],test_sample)
            distances.append(dist)
        category = distances.index(min(distances))
        predictions.append(category)
    
    return predictions
    

class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        Calculates self.X_mean_ based on the mean
        feature values in X for each class.
        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)
        fit always returns self.
        """
        #we make an array with the indexes of all labels
        indexes = [np.where(y==0)[0],np.where(y==1)[0],np.where(y==2)[0],np.where(y==3)[0],
           np.where(y==4)[0],np.where(y==5)[0],np.where(y==6)[0],np.where(y==7)[0],
           np.where(y==8)[0],np.where(y==9)[0]]
        
        self.X_mean_ = [np.zeros((16,16))]*10
        c = 0
        for j in indexes:
            sum = 0
            for i in j:
                temp = X[i].reshape(16,16)
                sum+=temp
            self.X_mean_[c] = sum/len(indexes[c])
            c+=1
        
        return self


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        return euclidean_distance_classifier(X, self.X_mean_)
    
    def score(self, X, y):
        s = 0
        predictions = self.predict(X)
        print("im in score")
        for j in range(len(y)):
            if predictions[j] == y[j]:
                s+=1
        return s/(len(y))

##########################################################################################################
########################################### step13 #######################################################
##########################################################################################################

#(a)
X = np.append(X_train, X_test).reshape(9298,256) #concantenate train and test set samples
y = np.append(y_train, y_test) #concantenate all the labels
#X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X, y, test_size=0.2, random_state=0)

#function that operates 5-fold-cross-validation
def return_mean_scores(Classifier,X,y):
    scores = cross_val_score(Classifier, X, y, 
                             cv=KFold(n_splits=5, random_state=42), 
                             scoring="accuracy")
    return np.mean(np.array(scores))


return_mean_scores(EuclideanDistanceClassifier(),X,y)


#(b)
X_new = X.reshape(9298,256)
pca = PCA(2)  # project from 256 to 2 dimensions
projected = pca.fit_transform(X_new)

matplotlib.pyplot.scatter(projected[:, 0], projected[:, 1],
            c = y, edgecolor='none', alpha=0.5,
            )
matplotlib.pyplot.xlabel('component 1')
matplotlib.pyplot.ylabel('component 2')
matplotlib.pyplot.colorbar();

#(c)
train_sizes, train_scores, test_scores = learning_curve(
    EuclideanDistanceClassifier(), X_new, y, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(.1, 1.0, 5))

def plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0, 1)):
    plt.figure()
    plt.title("Learning Curve")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(.6, 1))

def plot_clf(clf, X, y, labels):
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision surface of Classifier')
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                         np.arange(y_min, y_max, .05))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z).reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap=plt.get_cmap("tab10"), alpha=0.8)
    
    matplotlib.pyplot.scatter(X[:, 0], X[:, 1],
            c = y, edgecolor='none', alpha=0.8,cmap=plt.get_cmap("tab10")
            )
    #matplotlib.pyplot.xlabel('component 1')
    #matplotlib.pyplot.ylabel('component 2')
    matplotlib.pyplot.colorbar();
    
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()
   
clf = EuclideanDistanceClassifier()
clf.fit(projected, y)
plot_clf(clf, projected, y,y)   

##########################################################################################################
########################################### step14 #######################################################
##########################################################################################################

def calculate_priors(X, y):
    prior_probabilities = np.zeros(10)
    for c in y:
        prior_probabilities[c] += 1 
    prior_probabilities = prior_probabilities/len(y)
    return prior_probabilities # number of times a class occurs/ number of samples

##########################################################################################################
########################################### step15 #######################################################
##########################################################################################################

#(a)
class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance


    def fit(self, X, y):
        
        #we make an array with the indexes of all labels
        indexes = [np.where(y==0)[0],np.where(y==1)[0],np.where(y==2)[0],np.where(y==3)[0],
           np.where(y==4)[0],np.where(y==5)[0],np.where(y==6)[0],np.where(y==7)[0],
           np.where(y==8)[0],np.where(y==9)[0]]
        
        #compute the mean value for each feature of each class
        self.X_mean_ = [np.zeros((16,16))]*10
        c = 0
        for j in indexes:
            sum = 0
            for i in j:
                temp = X[i]
                sum+=temp
            self.X_mean_[c] = sum/len(indexes[c])
            c+=1
        
        #compute the varriance for each feature of each class
        var_values = [[]]*10
        for i in range(10):
            var = [0]*256
            all_features = [[]]*len(indexes[i])
            for j in range(len(indexes[i])):
                all_features[j] = X[indexes[i][j]]
            for k in range(256):
                var[k] = np.var(np.array(all_features)[:,k])
                
                ##########################################################################################################
                ########################################### step16 #######################################################
                ##########################################################################################################
                
                #var[k] = 1
            var_values[i] = var
        self.X_var_= var_values
        
        
        
        return self.X_mean_, self.X_var_


    def predict(self, X):
        """
        We can say that P(x=v|Ck)=f(x) where x is the features vector and f Gaussian
        distribution. We will use the varriance and the mean value of each feature from above
        f(x) = (1 / sqrt(2 * Pi) * sigma) * exp(-((x-mean)^2 / (2 * sigma^2)))
        
        conditional_probs (1x256)array: P(x=v|Ck)=f(x)
        product_probs (1x10)array: stores the product of conditional_probs elements
        class_probabilities (1x10)array: P(Ck)*{PRODUCT(P(x=v|Ck))} =  P(Ck)*product_probs
        
        the prediciton is the argmax of class_probabilities
        """
        MEAN = np.array(self.X_mean_)
        VAR = np.array(self.X_var_)
        priors = calculate_priors(X, y)
        predictions = np.zeros(len(X))

        for i in range(len(X)):
            conditional_probs = np.zeros(256)
            product_probs = np.zeros(10)
            class_probabilities = np.zeros(10)

            for j in range(10):
                VAR[j][VAR[j] == 0] = 1
                A = -(np.array(X[i])-MEAN[j])**2
                B = 2*(VAR[j]**2)
                C = np.sqrt(2*math.pi*VAR[j])
                D = A/B
                conditional_probs = (np.exp(D))/C

                
                product_probs = np.prod(conditional_probs, where = conditional_probs != 0)
                class_probabilities[j] = product_probs*priors[j]

            prediction = np.argmax(class_probabilities)
            predictions[i] = prediction
        return predictions
    
    def score(self, X, y):
        s = 0
        predictions = self.predict(X)
        for j in range(len(y)):
            if predictions[j] == y[j]:
                s+=1
        return s/(len(y))




X = np.append(X_train, X_test).reshape(9298,256)
y = np.append(y_train, y_test)

#(b)
#5-fold-cross-validation
return_mean_scores(CustomNBClassifier(),X,y)
## or just accuracy:
my_NBC = CustomNBClassifier()
my_NBC.fit(X_train, y_train)
my_NBC.score(X_test, y_test)


#(c)
gaussian = GaussianNB( var_smoothing=0.1)
#5-fold-cross-validation
return_mean_scores(gaussian,X,y)
## or just accuracy:
gaussian.fit(X_train, y_train)
gaussian.score(X_test, y_test)

##########################################################################################################
########################################### step16 #######################################################
##########################################################################################################

#there is the comment in the code above (lone 496)

##########################################################################################################
########################################### step17 #######################################################
##########################################################################################################


neighbors = KNeighborsClassifier(n_neighbors=3)
#5-fold-cross-validation
return_mean_scores(neighbors,X,y)

## or just accuracy:
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
neigh.score(X_test, y_test)



svc_linear = SVC(kernel="linear")  
return_mean_scores(svc_linear,X,y)

## or just accuracy:
svc_linear.fit(X_train, y_train)
svc_linear.score(X_test, y_test)


svc_rbf = SVC(probability=True, kernel="rbf")
#5-fold-cross-validation  
return_mean_scores(svc_rbf,X,y)

# or just accuracy:
svc_rbf.fit(X_train, y_train)
svc_rbf.score(X_test, y_test)

##########################################################################################################
########################################### step18 #######################################################
##########################################################################################################

#code for confusion matrix

def show_confusion_matrix(classifier, X_train, y_train, X_test, y_test, title):
    clf = classifier
    clf.fit(X_train, y_train) # fit your classifier

    # make predictions with your classifier
    y_pred = clf.predict(X_test) 

    M = confusion_matrix(y_test, y_pred)

    # plotting the confusion matrix
    plot_confusion_matrix(clf, X_test, y_test)
    plt.title(title)
    plt.show()
    
show_confusion_matrix(SVC(probability=True, kernel="linear"), X_train, y_train, X_test, y_test, "linear SVC Confusion Matrix")
show_confusion_matrix(SVC(probability=True, kernel="rbf"), X_train, y_train, X_test, y_test, "rbf kernel SVM Confusion Matrix")
show_confusion_matrix(GaussianNB(var_smoothing=0.1), X_train, y_train, X_test, y_test, "Gaussian Naive Bayes Confusion Matrix")
show_confusion_matrix(KNeighborsClassifier(n_neighbors=3), X_train, y_train, X_test, y_test, "K NeighborsClassifier Confusion Matrix")


eclf1 = VotingClassifier(estimators=[('svc_rbf', svc_rbf), ('neig', neigh), ('gnb', GaussianNB(var_smoothing=0.1))], voting='hard')
#5-fold-cross-validation 
return_mean_scores(eclf1,X,y)
# or just accuracy:
eclf1.fit(X_train, y_train)
eclf1.score(X_test, y_test)


eclf2 = VotingClassifier(estimators=[('svc_rbf', svc_rbf), ('neig', neigh), ('gnb', GaussianNB(var_smoothing=0.1))], voting='soft')
#5-fold-cross-validation 
return_mean_scores(eclf2,X,y)
# or just accuracy:
eclf2.fit(X_train, y_train)
eclf2.score(X_test, y_test)


bagging = BaggingClassifier(base_estimator=svc_rbf,
                        n_estimators=10, random_state=0)
#5-fold-cross-validation 
return_mean_scores(bagging,X,y)
# or just accuracy:
bagging.fit(X_train, y_train)
bagging.score(X_test, y_test)

#combination with decision tree in order to have Random Forest classification
tree = DecisionTreeClassifier(random_state=0)
random_forest = VotingClassifier(estimators=[('lr', bagging), ('rf', tree)], voting='hard')
#random_forest = RandomForestClassifier(max_depth=2, random_state=0)
#accuracy
random_forest.fit(X_train, y_train)
random_forest.score(X_test, y_test)
#5-fold-cross-validation
return_mean_scores(random_forest,X,y)

##########################################################################################################
########################################### step19 #######################################################
##########################################################################################################

#(a)

# epochs = the number of times we are going to feed the whole dataset to the network
EPOCHS = 30 # more epochs means more training on the given data. IS this good ?? 60
learning_rate = 1e-2
# the mini-batch size, usually a power of 2 but not restrictive rule in general
BATCH_SZ = 64

class DigitsData(Dataset):
    def __init__(self, X, y, trans=None):
        # all the available data are stored in a list
        self.data = list(zip(X, y))
        # we optionally may add a transformation on top of the given data
        # this is called augmentation in realistic setups
        self.trans = trans
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.trans is not None:
            return self.trans(self.data[idx])
        else:
            return self.data[idx]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
train_data = DigitsData(X_train, y_train, trans=None)
test_data = DigitsData(X_test, y_test, trans=None)
        
train_dl = DataLoader(train_data, batch_size=BATCH_SZ, shuffle=True)


#(b)
# alternative definitions of neural network models in PyTorch

# definition-1: compact but not generic
in_feats = 256
out_feats = 10
linearw_activation_model = nn.Sequential(
    nn.Linear(in_feats, out_feats),
    nn.ReLU()
)

# definition-2: The proposed way to define NNs in PyTorch 

# always inherit from nn.Module
class LinearWActivation(nn.Module): 
    def __init__(self, in_features, out_features, activation='sigmoid'):
        super(LinearWActivation, self).__init__()
        # nn.Linear is just a matrix of [in_features, out_features] randomly initialized
        self.f = nn.Linear(in_features, out_features)
        if activation == 'sigmoid':
            self.a = nn.Sigmoid()
        else:
            self.a = nn.ReLU()
      
      # this would also do the job
      # self.t = nn.Sequenntial(self.f, self. a)
          
  # the forward pass of info through the net
    def forward(self, x): 
        return self.a(self.f(x))




# again we inherit from nn.Module
class DigitNet(nn.Module): 
    def __init__(self, layers, n_features, n_classes, activation='sigmoid'):
        '''
      Args:
        layers (list): a list of the number of consecutive layers
        n_features (int):  the number of input features
        n_classes (int): the number of output classes
        activation (str): type of non-linearity to be used
          '''
        super(DigitNet, self).__init__()
        layers_in = [n_features] + layers # list concatenation
        layers_out = layers + [n_classes]
        # loop through layers_in and layers_out lists
        self.f = nn.Sequential(*[
            LinearWActivation(in_feats, out_feats, activation=activation)
            for in_feats, out_feats in zip(layers_in, layers_out)])
        # final classification layer is always a linear mapping
        self.clf = nn.Linear(n_classes, n_classes)
                
    def forward(self, x): # again the forwrad pass
      # apply non-linear composition of layers/functions
      y = self.f(x)
      # return an affine transformation of y <-> classification layer
      return self.clf(y)


model = DigitNet([100]*256, X.shape[1], len(set(y)))
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-2  # the ETA variable in gradient descent
optimizer = optim.SGD(model.parameters(), lr=learning_rate) # feed the optimizer with the netowrk parameters




model.train() # gradients "on"
for epoch in range(EPOCHS): # loop through dataset
    running_average_loss = 0
    for i, data in enumerate(train_dl): # loop thorugh batches
        X_batch, y_batch = data # get the features and labels
        optimizer.zero_grad() # ALWAYS USE THIS!! 
        out = model(X_batch.float()) # forward pass
        loss = criterion(out, y_batch.long()) # compute per batch loss 
        loss.backward() # compurte gradients based on the loss function
        optimizer.step() # update weights 
        
        running_average_loss += loss.detach().item()
        if i % 100 == 0:
            print("Epoch: {} \t Batch: {} \t Loss {}".format(epoch, i, float(running_average_loss) / (i + 1)))




model.eval() # turns off batchnorm/dropout ...
acc = 0
n_samples = 0
with torch.no_grad(): # no gradients required!! eval mode, speeds up computation
    for i, data in enumerate(test_dl):
        X_batch, y_batch = data # test data and labels
        out = net(X_batch.float()) # get net's predictions
        val, y_pred = out.max(1) # argmax since output is a prob distribution
        acc += (y_batch.long() == y_pred).sum().detach().item() # get accuracy
        n_samples += X_batch.size(0)


print(acc)
print(n_samples)

print(acc / n_samples)


#(c)


class PytorchNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self):
        # WARNING: Make sure predict returns the expected (nsamples) numpy array not a torch tensor.
        # TODO: initialize model, criterion and optimizer
        self.model = DigitNet([100]*256, X.shape[1], len(set(y)))
        self.criterion = nn.CrossEntropyLoss()
        

    def fit(self, X, y,BATCH_SZ,EPOCHS,learning_rate):
        # TODO: split X, y in train and validation set and wrap in pytorch dataloaders
        train_data = DigitsData(X, y, trans=None)
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate) # feed the optimizer with the netowrk parameters

        
        train_dl = DataLoader(train_data, batch_size=BATCH_SZ, shuffle=True)
        model.train() # gradients "on"
        
        for epoch in range(EPOCHS): # loop through dataset
            running_average_loss = 0
            for i, data in enumerate(train_dl): # loop thorugh batches
                X_batch, y_batch = data # get the features and labels
                optimizer.zero_grad() # ALWAYS USE THIS!! 
                out = model(X_batch.float()) # forward pass
                loss = criterion(out, y_batch.long()) # compute per batch loss 
                loss.backward() # compurte gradients based on the loss function
                optimizer.step() # update weights 

                running_average_loss += loss.detach().item()
                
    def predict(self, X):
        # TODO: wrap X in a test loader and evaluate
        test_data = DigitsData(X, y, trans=None)
        test_dl = DataLoader(test_data, batch_size=BATCH_SZ, shuffle=True)
        
        model.eval() # turns off batchnorm/dropout ...
        acc = 0
        n_samples = 0
        predictions = []
        with torch.no_grad(): # no gradients required!! eval mode, speeds up computation
            for i, data in enumerate(test_dl):
                X_batch, y_batch = data # test data and labels
                out = model(X_batch.float()) # get net's predictions
                val, y_pred = out.max(1) # argmax since output is a prob distribution
                predictions.append(y_pred)
                """
                acc += (y_batch.long() == y_pred).sum().detach().item() # get accuracy
                n_samples += X_batch.size(0)
            self.n_samples = n_samples
                """
            self.predictions = predictions
            #return self.n_samples
        return self.predictions
            
    def score(self, X, y):
        test_data = DigitsData(X, y, trans=None)
        test_dl = DataLoader(test_data, batch_size=BATCH_SZ, shuffle=True)
        acc = 0
        n_samples = 0
        
        with torch.no_grad(): # no gradients required!! eval mode, speeds up computation
            for i, data in enumerate(test_dl):
                X_batch, y_batch = data # test data and labels
                out = model(X_batch.float()) # get net's predictions
                
                
                acc += (y_batch.long() == self.predictions[i]).sum().detach().item() # get accuracy
                n_samples += X_batch.size(0)
            self.n_samples = n_samples
                
            self.predictions = predictions
            #return self.n_samples
            #return self.predictions
        return acc / self.n_samples


#accuracy
NNM= PytorchNNModel()
NNM.fit(X_train,y_train,BATCH_SZ,EPOCHS,learning_rate) # the ETA variable in gradient descent
NNM.predict(X_test)
NNM.score(X_test,y_test)





