#!/usr/bin/env python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import librosa.display
import matplotlib.pyplot as plt
import copy
import os
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from collections import Counter
import torch
from torch import nn
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

###################################################
##################### STEP 0 ######################
###################################################


os.listdir("/kaggle/input/patreco3-multitask-affective-music/data")
#we activate and deactivate gpu from settings --> accelerator --> GPU


###################################################
##################### STEP 1 ######################
###################################################

############### a) ################################

#Load a sample file
jazz_sample = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train/10975.fused.full.npy')
electronic_sample = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train/29026.fused.full.npy')
print("the shape of jazz sample is: ", jazz_sample.shape)
print("the shape of electronic sample is: ", electronic_sample.shape)


############# b) ##################################

jazz_mel = jazz_sample[:128]
electronic_mel = electronic_sample[:128]


############ c) ###################################



# Plot the spectrograms

fig, ax = plt.subplots()
img = librosa.display.specshow(jazz_mel, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Jazz Spectrogram ID: 10975')
fig.colorbar(img, ax=ax, format="%+2.f dB")

fig, ax = plt.subplots()
img = librosa.display.specshow(electronic_mel, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Electronic Spectrogram ID: 29026')
fig.colorbar(img, ax=ax, format="%+2.f dB")


###################################################
##################### STEP 2 ######################
###################################################

################## a) #############################

print("the shape of jazz spectogramm is: ", jazz_mel.shape)
print("the shape of electronic spectogramm is: ", electronic_mel.shape)


################# b) ##############################

jazz_med_sample = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train/10975.fused.full.npy')
electronic_med_sample = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train/29026.fused.full.npy')

jazz_med = jazz_med_sample[:128]
electronic_med = electronic_med_sample[:128]

print("the shape of jazz beat sample is: ", jazz_med.shape)
print("the shape of electronic beat sample is: ", electronic_med.shape)

# Plot the spectrograms

fig, ax = plt.subplots()
img = librosa.display.specshow(jazz_med, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Jazz median Spectrogram ID: 10975')
fig.colorbar(img, ax=ax, format="%+2.f dB")

fig, ax = plt.subplots()
img = librosa.display.specshow(electronic_med, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Electronic median Spectrogram ID: 29026')
fig.colorbar(img, ax=ax, format="%+2.f dB")


###################################################
##################### STEP 3 ######################
###################################################


jazz_chroma = jazz_sample[128:]
electronic_chroma = electronic_sample[128:]

# Plot the spectrograms

fig, ax = plt.subplots()
img = librosa.display.specshow(jazz_chroma, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Jazz Chromagram ID: 10975')
fig.colorbar(img, ax=ax, format="%+2.f dB")

fig, ax = plt.subplots()
img = librosa.display.specshow(electronic_chroma, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Electronic Chromagram ID: 29026')
fig.colorbar(img, ax=ax, format="%+2.f dB")

print("the shape of jazz Chromagram is: ", jazz_chroma.shape)
print("the shape of electronic Chromagram is: ", electronic_chroma.shape)


jazz_med_sample = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train/10975.fused.full.npy')
electronic_med_sample = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train/29026.fused.full.npy')


jazz_med_chroma = jazz_med_sample[128:]
electronic_med_chroma = electronic_med_sample[128:]

print("the shape of jazz beat sample is: ", jazz_med_chroma.shape)
print("the shape of electronic beat sample is: ", electronic_med_chroma.shape)

# Plot the spectrograms

fig, ax = plt.subplots()
img = librosa.display.specshow(jazz_med_chroma, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Jazz median Spectrogram ID: 10975')
fig.colorbar(img, ax=ax, format="%+2.f dB")

fig, ax = plt.subplots()
img = librosa.display.specshow(electronic_med_chroma, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Electronic median Spectrogram ID: 29026')
fig.colorbar(img, ax=ax, format="%+2.f dB")


###################################################
##################### STEP 4 ######################
###################################################

#################### a) ###########################


# HINT: Use this class mapping to merge similar classes and ignore classes that do not work very well
CLASS_MAPPING = {
    "Rock": "Rock",
    "Psych-Rock": "Rock",
    "Indie-Rock": None,
    "Post-Rock": "Rock",
    "Psych-Folk": "Folk",
    "Folk": "Folk",
    "Metal": "Metal",
    "Punk": "Metal",
    "Post-Punk": None,
    "Trip-Hop": "Trip-Hop",
    "Pop": "Pop",
    "Electronic": "Electronic",
    "Hip-Hop": "Hip-Hop",
    "Classical": "Classical",
    "Blues": "Blues",
    "Chiptune": "Electronic",
    "Jazz": "Jazz",
    "Soundtrack": None,
    "International": None,
    "Old-Time": None,
}


def torch_train_val_split(
    dataset, batch_train, batch_eval, val_size=0.2, shuffle=True, seed=420
):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_train, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_eval, sampler=val_sampler)
    return train_loader, val_loader


def read_spectrogram(spectrogram_file, chroma='mel'):
    spectrograms = np.load(spectrogram_file)
    if chroma == 'chroma':
        spectrograms = spectrograms[128:]
    elif chroma=='mel':
        spectrograms = spectrograms[:128]        
    return spectrograms.T


class LabelTransformer(LabelEncoder):
    def inverse(self, y):
        try:
            return super(LabelTransformer, self).inverse_transform(y)
        except:
            return super(LabelTransformer, self).inverse_transform([y])

    def transform(self, y):
        try:
            return super(LabelTransformer, self).transform(y)
        except:
            return super(LabelTransformer, self).transform([y])


class PaddingTransform(object):
    def __init__(self, max_length, padding_value=0):
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, s):
        if len(s) == self.max_length:
            return s

        if len(s) > self.max_length:
            return s[: self.max_length]

        if len(s) < self.max_length:
            s1 = copy.deepcopy(s)
            pad = np.zeros((self.max_length - s.shape[0], s.shape[1]), dtype=np.float32)
            s1 = np.vstack((s1, pad))
            return s1


class SpectrogramDataset(Dataset):
    def __init__(
        self, path, class_mapping=None, train=True, max_length=-1, regression=None, chroma = 'mel'):
        t = "train" if train else "test"
        p = os.path.join(path, t)
        self.regression = regression
        
        self.chroma = chroma
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        self.files, labels = self.get_files_labels(self.index, class_mapping)
        self.feats = [read_spectrogram(os.path.join(p, f), self.chroma) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        self.label_transformer = LabelTransformer()
        if isinstance(labels, (list, tuple)):
            if not regression:
                self.labels = np.array(
                    self.label_transformer.fit_transform(labels)
                ).astype("int64")
            else:
                self.labels = np.array(labels).astype("float64")

    def get_files_labels(self, txt, class_mapping):
        with open(txt, "r") as fd:
            lines = [l.rstrip().split("\t") for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            if self.regression:
                l = l[0].split(",")
                files.append(l[0] + ".fused.full.npy")
                labels.append(l[self.regression])
                continue
            label = l[1]
            if class_mapping:
                label = class_mapping[l[1]]
            if not label:
                continue
            fname = l[0]
            _id = fname.split('.')[0]
            npy_file = '{}.fused.full.npy'.format(_id)
            files.append(npy_file)
            labels.append(label)
            
            if fname.endswith(".gz"):
                fname = ".".join(fname.split(".")[:-1])
#             files.append(fname)
#             labels.append(label)
        return files, labels

    def __getitem__(self, item):
        length = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], length

    def __len__(self):
        return len(self.labels)


# if __name__ == "__main__":

#     dataset = SpectrogramDataset(
#         "/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms", class_mapping=CLASS_MAPPING, train=True
#     )
    
# #     dataset = SpectrogramDataset(
# #         "/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms", class_mapping=None, train=True
# #     )

#     print(dataset[10])
#     print(f"Input: {dataset[10][0].shape}")
#     print(f"Label: {dataset[10][1]}")
#     print(f"Original length: {dataset[10][2]}")
#     #print(len(dataset))


############## c) ###############################

dataframe = pd.read_csv("/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train_labels.txt", sep='\t')
all_genres = dataframe["Genre"].tolist()

genres_dict = Counter(all_genres)

genres = []
values = []

for k, v in genres_dict.items():
    genres.append(k)
    values.append(v)

fig = plt.figure(figsize=(17, 5))
ax = fig.add_subplot(1,1,1)
plt.bar(genres, values)
plt.grid()
plt.xticks(genres,genres,rotation=90)
plt.title('Initial Spectrogram')
plt.xlabel('Class')
plt.ylabel('Number')
plt.grid()
plt.show()    


dataset = SpectrogramDataset(
       path="/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms", class_mapping=CLASS_MAPPING, 
       train=True,max_length=-1, regression=None, chroma = 'mel')


merged_labels = dataset.labels

merged_labels_dict = Counter(merged_labels)

genres = []
values = []

for k, v in merged_labels_dict.items():
    genres.append(k)
    values.append(v)

genres = dataset.label_transformer.inverse(genres)

fig = plt.figure(figsize=(17, 5))
ax = fig.add_subplot(1,1,1)
plt.bar(genres, values)
plt.grid()
plt.xticks(genres,genres,rotation=90)
plt.title('Histogram after ignoring and merging')
plt.xlabel('Class')
plt.ylabel('Number')
plt.grid()
plt.show()    


###################################################
##################### STEP 5 ######################
###################################################

####################################################
#### BASED ON: https://cnvrg.io/pytorch-lstm/ ######
####################################################

device = torch.device("cuda")
print(device)


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
        #fully connected last layer
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
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
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


###################### BASED ON THE TUTORIALS: #########################
##### https://pytorch.org/tutorials/beginner/introyt/trainingyt.html ###
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html ###
########################################################################



def train_model(mel_specs, rnn_size,num_layers, epochs, batch_size, learning_rate,title, bidirectional = False, dropout = 0.0, regularization = 0.0, early_stopping = False, overfit_batch = False):
      
    #dropout and regularization parameter is for ex 14.5
    #early_stopping parameter is for ex 14.6
    #bidirectional parameter is for ex 16.7
    
    #load our model with the function we created above
    model = BasicLSTM(input_dim = len(list(mel_specs)[0][0][0]), rnn_size = rnn_size, output_dim = 10, num_layers = num_layers, bidirectional = bidirectional, dropout = dropout)
    #load model to GPU in order to save time
    model.to(device)
    

    #criterion:we try to find the parameters that maximize the probability of the training data
    criterion = nn.CrossEntropyLoss()
    #Optimization algorithm changes the attributes of the neural network such as weights  
    #to reduce the losses. Optimizers are used to solve optimization problems by minimizing the function.
    optimizer = torch.optim.Adam(model.parameters(), weight_decay = regularization, lr = learning_rate)
    training_loss = []
    validation_loss = []
   
    #we set this as default value
    val_size = 0.15
    
    if overfit_batch:
        batch_size = 3
    train_data, validation_data = torch_train_val_split(mel_specs, batch_size ,batch_size, val_size)

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
    
    train_batch_number =  int(0.85*len(mel_specs.labels) / batch_size))
    val_batch_number = int(0.15*len(mel_specs.labels)/batch_size)
    
    # training, calclulating loss for train / val sets
    for epoch in range(epochs):
        trloss = 0 
        valoss = 0
        
        times = 0
        if overfit_batch:
            #we take the first four batces
            #each batch contains four samples
            for data in train_data:
                if times == 4:
                    break
                else:
                    ################ get the inputs #######################
                    X_train_batch = data[0].float().cuda()
                    y_train_batch = data[1].cuda()
                    lengths = data[2].cuda()
                    y_pred = model(X_train_batch,lengths)
                    ############# zero the parameter gradients ###########
                    optimizer.zero_grad()
                    ############# forward + backward + optimize ##########
                    #https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#
                    loss = criterion(y_pred, y_train_batch)
                    loss.backward()
                    optimizer.step()
                    trloss +=  loss
                    times+=1
        else:
            for data in train_data:
                ################ get the inputs #######################
                X_train_batch = data[0].float().cuda()
                y_train_batch = data[1].cuda()
                lengths = data[2].cuda()
                y_pred = model(X_train_batch,lengths)
                ############# zero the parameter gradients ###########
                optimizer.zero_grad()
                ############# forward + backward + optimize ##########
                #https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#
                loss = criterion(y_pred, y_train_batch)
                loss.backward()
                optimizer.step()
                trloss +=  loss

        # append train loss
        ###### print every len(X_train)/ batch_size mini-batches
        training_loss.append(trloss.item()/train_batch_number)
        print('Train loss for epoch ',epoch+1,":",trloss.item()/train_batch_number)

        # validation set
        valoss = 0
        with torch.no_grad():
            model.eval()
            
            for val_data in validation_data:
                X_dev_batch = val_data[0].float().cuda() 
                y_dev_batch = val_data[1].cuda()
                lengths = val_data[2].cuda()
                #print("the lengths are: ",lengths.size())
                #print("X_dev_batch is: ", X_dev_batch.size())
                y_dev_pred = model(X_dev_batch,lengths)
                loss = criterion(y_dev_pred,y_dev_batch)
                valoss +=  loss.detach().item()

        ########################## 14.6 #############################
        ########## this is our early stopping condition #############
        #### if early stopping is activated the we do the checkpoint#
        ############################################################# 
                if early_stopping == True and epoch > 10:
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
                if valoss < best_validation_score:
                    best_validation_score = valoss
                    best_validation_model = copy.deepcopy(model)
                    torch.save(model,'best') 
            
            
            validation_loss.append(valoss/val_batch_number)
            print('Development loss for epoch ',epoch+1,":",valoss/val_batch_number)
            model.train()
    
    epoch_linspace = np.linspace(1,epoch+1,epoch+1)    
    ################### Loss/Epoch diagram for train and val dataset ##########
    plt.plot(epoch_linspace,training_loss,'royalblue',label = 'Training loss')
    plt.plot(epoch_linspace,validation_loss,'moccasin',label = 'Validation loss')
    plt.title(title)
    plt.legend(loc = 'best')
    plt.show()
    return model


################## b) ###################################

val_size = 0.15
mel_specs_overfit_batch = SpectrogramDataset(path='/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',class_mapping=CLASS_MAPPING,train=True,max_length=-1, regression=None, chroma = 'mel')
spec_model = train_model(mel_specs_overfit_batch, rnn_size=64,num_layers=3,epochs=30,batch_size=16,learning_rate=0.0001, title = "Bidirectional model",bidirectional = True, dropout=0.2,regularization=0.4,early_stopping = False, overfit_batch=True)


################## c) ###################################

mel_specs = SpectrogramDataset(path='/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',class_mapping=CLASS_MAPPING,train=True,max_length=-1, regression=None, chroma = 'mel')
spec_model = train_model(mel_specs, rnn_size=64,num_layers=3,epochs=30,batch_size=16,learning_rate=0.0001, title = "Bidirectional model",bidirectional = True, dropout=0.2,regularization=0.4,early_stopping = False, overfit_batch=False)


################## d) ###################################

beat_mels = SpectrogramDataset(path='/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',class_mapping=CLASS_MAPPING,train=True,max_length=-1, regression=None, chroma = 'mel')
beat_model = train_model(beat_mels, rnn_size=64,num_layers=3,epochs=30,batch_size=16,learning_rate=0.0001, title = "Bidirectional beat model",bidirectional = True, dropout=0.2,regularization=0.4,early_stopping = False, overfit_batch=False)


################## e) ###################################

beat_chromas = SpectrogramDataset(path='/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',class_mapping=CLASS_MAPPING,train=True,max_length=-1, regression=None, chroma = 'chroma')
beat_model_chromas = train_model(beat_chromas, rnn_size=64,num_layers=3,epochs=30,batch_size=16,learning_rate=0.0001, title = "Bidirectional beat model",bidirectional = True,dropout=0.2,regularization=0.4,early_stopping = False, overfit_batch=False)    


################## f) ###################################

beat_mel_chromas = SpectrogramDataset(path='/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',class_mapping=CLASS_MAPPING,train=True,max_length=-1, regression=None, chroma = 'both')
beat_model_mel_chromas = train_model(beat_mel_chromas, rnn_size=64,num_layers=3,epochs=30,batch_size=16,learning_rate=0.0001, title = "Bidirectional beat mel and chroma model",bidirectional = True,dropout=0.2,regularization=0.4,early_stopping = False, overfit_batch=False)    


###################################################
##################### STEP 6 ######################
###################################################



def test_model(model, test_dataset,batch_size):
        
    test_data = DataLoader(test_dataset, batch_size=batch_size)
    model.eval()
    
    predictions, labels = [], []
    
    for t_data in test_data:
             
            X_test_batch = t_data[0].float().cuda() 
            y_test_batch = t_data[1].cuda()
            lengths = t_data[2].cuda()

            output = model(X_test_batch, lengths) 
            
            y_pred = torch.nn.functional.softmax(output, dim = -1).argmax(dim = -1)
            
            y_pred = y_pred.data.cpu().numpy()
            y_test_batch = y_test_batch.data.cpu().numpy()

            predictions.append(y_pred)
            labels.append(y_test_batch)
            
    print(classification_report(labels, predictions))
    
    return  

###########################################################
############## test each model from above #################
###########################################################

batch_train = 14
test_model(spec_model, mel_specs_overfit_batch ,batch_train)


batch_train = 14
test_model(spec_model, mel_specs ,batch_train)


batch_train = 14
test_model(beat_model, beat_mels ,batch_train)


batch_train = 14
test_model(beat_model_chromas, beat_chromas ,batch_train)


batch_train = 14
test_model(beat_model_mel_chromas, beat_mel_chromas ,batch_train)

class BasicLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, bidirectional=False, dropout = 0, regression= False):
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
        #fully connected last layer
        self.fc = nn.Linear(3, self.output_dim)
        self.relu = nn.ReLU()
        self.linear_regression = nn.Linear(self.output_dim, 1)
        self.regression = regression
    
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
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
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
        last_outputs = self.fc(out)
        if self.regression:
            last_outputs =  self.linear_regression(last_outputs)
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



# In[6]:


import warnings    


def train_model(mel_specs, rnn_size,num_layers, epochs, batch_size, learning_rate,title, bidirectional = False, dropout = 0.0, regularization = 0.0, early_stopping = False, overfit_batch = False, regression = False):
      
    #dropout and regularization parameter is for ex 14.5
    #early_stopping parameter is for ex 14.6
    #bidirectional parameter is for ex 16.7
    warnings.filterwarnings('ignore')
    #load our model with the function we created above
    model = BasicLSTM(input_dim = len(list(mel_specs)[0][0][0]), rnn_size = rnn_size, output_dim = 10, num_layers = num_layers, bidirectional = bidirectional, dropout = dropout, regression = regression)
    #load model to GPU in order to save time
    model.to(device)
    

    #criterion:we try to find the parameters that maximize the probability of the training data
    if regression:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    #Optimization algorithm changes the attributes of the neural network such as weights  
    #to reduce the losses. Optimizers are used to solve optimization problems by minimizing the function.
    optimizer = torch.optim.Adam(model.parameters(), weight_decay = regularization, lr = learning_rate)
    training_loss = []
    validation_loss = []
    val_size = 0.15
    if overfit_batch:
        batch_size = 3
    train_size = int(0.85 * len(mel_specs))
    train_loader, val_loader = torch.utils.data.random_split(mel_specs,[int(0.85 * len(mel_specs)), len(mel_specs) - train_size])
    train_loader = DataLoader(train_loader, batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_loader, batch_size=batch_size,shuffle=True)
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
        
        times = 0
        if overfit_batch:
            #we take the first four batces
            #each batch contains four samples
            for data in train_loader:
                if times == 4:
                    break
                else:
                    ################ get the inputs #######################
                    X_train_batch = data[0].float().cuda()
                    y_train_batch = data[1].cuda()
                    lengths = data[2].cuda()
                    y_pred = model(X_train_batch,lengths)
                    ############# zero the parameter gradients ###########
                    optimizer.zero_grad()
                    ############# forward + backward + optimize ##########
                    #https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#
                    loss = criterion(y_pred, y_train_batch)
                    loss.backward()
                    optimizer.step()
                    trloss +=  loss
                    times+=1
        else:
            for data in train_loader:
                ################ get the inputs #######################
                X_train_batch = data[0].float().cuda()
                y_train_batch = data[1].cuda()
                lengths = data[2].cuda()
                y_pred = model(X_train_batch,lengths)
                ############# zero the parameter gradients ###########
                optimizer.zero_grad()
                ############# forward + backward + optimize ##########
                #https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#
                if regression:
                    y_pred = y_pred.to(torch.float32)
                    y_train_batch = y_train_batch.to(torch.float32)
                loss = criterion(y_pred, y_train_batch)
                loss.backward()
                optimizer.step()
                trloss +=  loss

        # append train loss
        ###### print every len(X_train)/ batch_size mini-batches
        training_loss.append(trloss.item()/(int(0.85*len(mel_specs.labels) / batch_size)))
        print('Train loss for epoch ',epoch+1,":",trloss.item()/(int(0.85*len(mel_specs.labels) / batch_size)))

        # validation set
        valoss = 0
        with torch.no_grad():
            model.eval()
            
            for val_data in val_loader:
                X_dev_batch = val_data[0].float().cuda() 
                y_dev_batch = val_data[1].cuda()
                lengths = val_data[2].cuda()
                y_dev_pred = model(X_dev_batch,lengths)
                loss = criterion(y_dev_pred,y_dev_batch)
                valoss +=  loss.detach().item()

#         ########################## 14.6 #############################
#         ########## this is our early stopping condition #############
#         #### if early stopping is activated the we do the checkpoint#
#         ############################################################# 
#                 if early_stopping == True and epoch > 10:
#                     if early > valoss.item():
#                         early = valoss.item()
#                     else:
#                         stopped_break = epoch
#                         stop_training = True
#                         break
#                 if stop_training == True:
#                     break

#         ########## we save the model with the minimum loss #########
#         ########## in validation set using torch.save ##############
#         ############################################################
                if valoss < best_validation_score:
                    best_validation_score = valoss
                    best_validation_model = copy.deepcopy(model)
                    torch.save(model,'best') 
            
            v_den = int(0.15*len(mel_specs.labels)/batch_size)
            validation_loss.append(valoss/v_den)
            print('Development loss for epoch ',epoch+1,":",valoss/v_den)
            model.train()
    
    epoch_linspace = np.linspace(1,epoch+1,epoch+1)    
    ################### Loss/Epoch diagram for train and val dataset ##########
    plt.plot(epoch_linspace,training_loss,'royalblue',label = 'Training loss')
    plt.plot(epoch_linspace,validation_loss,'moccasin',label = 'Validation loss')
    plt.title(title)
    plt.legend(loc = 'best')
    plt.show()
    return model, val_loader


# In[7]:


from sklearn.metrics import classification_report

def test_model(model, test_dataset,batch_train):
        
    #test_loader = DataLoader(test_dataset, batch_size=batch_train)
    model.eval()
    
    y_predictions,y_real = [], []
    
    for test_data in test_dataset:
             
            X_test_batch = test_data[0].float().cuda() 
            y_test_batch = test_data[1].cuda()
            lengths = test_data[2].cuda()
            
            y_pred = model(X_test_batch, lengths) # feed-forward
            
            # softmax at the last layer
            #y_pred = torch.nn.functional.softmax(out, dim = -1).argmax(dim = -1)
            
            #append results 
            #if is_cuda:
            y_pred = y_pred.data.cpu().numpy().tolist()
            y_test_batch = y_test_batch.data.cpu().numpy().tolist()

            y_predictions += y_pred
            y_real += y_test_batch
    #print(classification_report(y_real, y_predictions))
    print("Spearman's Correlation is:", scipy.stats.spearmanr(y_real, y_predictions))
    return  

###################################################
##################### STEP 7 ######################
###################################################


class Cnn(nn.Module):
    def __init__(self, regression = False):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.batch_normal1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.batch_normal2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.batch_normal3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.batch_normal4 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(256*19, 100)
        self.fc2 = nn.Linear(100, 10)
        #self.fc2_regression = nn.Linear(100, 1)
        
        ### we change it just for step 10 ###
        self.fc2_regression = nn.Linear(100, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(4)
        self.regression = regression
    # x represents our data
    def forward(self, x):
        
        #first layer
        x = self.conv1(x)
        x = self.batch_normal1(x)
        x = F.relu(x)
        x = self.pool1(x)
        #x = self.dropout(x)
        
        #second layer
        x = self.conv2(x)
        x = self.batch_normal2(x)
        x = F.relu(x)
        x = self.pool1(x)
        #x = self.dropout(x)
        
        #third layer
        x = self.conv3(x)
        x = self.batch_normal3(x)
        x = F.relu(x)
        x = self.pool2(x)
        #x = self.dropout(x)
        
        #fourth layer
        x = self.conv4(x)
        x = self.batch_normal4(x)
        x = F.relu(x)
        x = self.pool2(x)
        #x = self.dropout(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        ######## just for step 10 we change it!!! #######
        if self.regression:
            x = self.fc2_regression(x)
            return x
        else:
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            #output = F.log_softmax(x, dim=3)
            return output




def train_CNN_model(model, mel_specs, epochs, batch_size, learning_rate,title, early_stopping = False, overfit_batch = False, regression = False):
#def train_CNN_model(mel_specs, epochs, batch_size, learning_rate,title, early_stopping = False, overfit_batch = False, regression = False):
      
    
    #model = Cnn(regression)
    device = torch.device("cuda")
    model.to(device)
    
    if regression:
        #criterion = nn.MSELoss()
        ############# just for step 10 #########
        criterion = my_loss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay = 0.4, lr = learning_rate)
    training_loss = []
    validation_loss = []
    val_size = 0.15
    
    
    if overfit_batch:
        batch_size = 3
    #only for step 7 and 9a in training the first model
    #train_loader, val_loader = torch_train_val_split(mel_specs, batch_size ,batch_size, val_size)
    
    #step 8
    train_size = int(0.85 * len(mel_specs))
    train_loader, val_loader = torch.utils.data.random_split(mel_specs,[int(0.85 * len(mel_specs)), len(mel_specs) - train_size])
    train_loader = DataLoader(train_loader, batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_loader, batch_size=batch_size,shuffle=True)
    
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
        
        times = 0
        if overfit_batch:
            #we take the first four batces
            #each batch contains four samples
            for data in train_loader:
                
                if times == 4:
                    break
                else:
                    count = 0
                    ################ get the inputs #######################
                    X_train_batch = data[0].float().cuda().view(-1,1,1293,len(list(mel_specs)[0][0][0]))
                    y_train_batch = data[1].cuda()
                    lengths = data[2].cuda()
                    y_pred = model(X_train_batch)
                    ############# zero the parameter gradients ###########
                    optimizer.zero_grad()
                    ############# forward + backward + optimize ##########
                    #https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#
                    loss = criterion(y_pred, y_train_batch)
                    loss.backward()
                    optimizer.step()
                    trloss +=  loss
                    count +=1
                    times+=1
        else:
            count = 0
            for data in train_loader:
                ################ get the inputs #######################
                X_train_batch = data[0].float().cuda().view(-1,1,1293,len(list(mel_specs)[0][0][0]))
                y_train_batch = data[1].cuda()
                lengths = data[2].cuda()
                y_pred = model(X_train_batch)
                ############# zero the parameter gradients ###########
                optimizer.zero_grad()
                ############# forward + backward + optimize ##########
                #https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#
                if regression:
                    y_pred = y_pred.to(torch.float32)
                    y_train_batch = y_train_batch.to(torch.float32)
                loss = criterion(y_pred, y_train_batch)
                loss.backward()
                optimizer.step()
                trloss +=  loss
                count +=1
        # append train loss
        ###### print every len(X_train)/ batch_size mini-batches
        
        #steps 7,8,9
        #training_loss.append(trloss.item()/(int(0.85*len(mel_specs.labels) / batch_size)))
        
        ############### just for step 10######################################################
        training_loss.append(trloss.item()/count)
        print('Train loss for epoch ',epoch+1,":",trloss.item()/count)
        #print('Train loss for epoch ',epoch+1,":",trloss.item()/(int(0.85*len(mel_specs.labels) / batch_size)))

        # validation set
        valoss = 0
        
        with torch.no_grad():
            model.eval()
            count1 = 0
            for val_data in val_loader:
                X_dev_batch = val_data[0].float().cuda().view(-1,1,1293,len(list(mel_specs)[0][0][0]))
                y_dev_batch = val_data[1].cuda()
                lengths = val_data[2].cuda()
                y_dev_pred = model(X_dev_batch)
                loss = criterion(y_dev_pred,y_dev_batch)
                valoss +=  loss.detach().item()

        
#                 if early_stopping == True and epoch > 10:
#                     if early > valoss.item():
#                         early = valoss.item()
#                     else:
#                         stopped_break = epoch
#                         stop_training = True
#                         break
#                 if stop_training == True:
#                     break

                if valoss < best_validation_score:
                    best_validation_score = valoss
                    best_validation_model = copy.deepcopy(model)
                    torch.save(model,'best') 
                count1 += 1
            
            #v_den = int(0.15*len(mel_specs.labels)/batch_size)
            #validation_loss.append(valoss/v_den)
            ########## step 10############
            validation_loss.append(valoss/count1)
            print('Development loss for epoch ',epoch+1,":",valoss/count1)
            
            #print('Development loss for epoch ',epoch+1,":",valoss/v_den)
            
            model.train()
    
    epoch_linspace = np.linspace(1,epoch+1,epoch+1)    
    ################### Loss/Epoch diagram for train and val dataset ##########
    plt.plot(epoch_linspace,training_loss,'royalblue',label = 'Training loss')
    plt.plot(epoch_linspace,validation_loss,'moccasin',label = 'Validation loss')
    plt.title(title)
    plt.legend(loc = 'best')
    plt.show()
    #step 7 and step 9 in training the first model
    #return model
    #step 8 
    return model, val_loader


device = torch.device("cuda")
print(device)




val_size = 0.15
mel_specs = SpectrogramDataset(path='/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',class_mapping=CLASS_MAPPING,train=True,max_length=-1, regression=None, chroma = 'mel')
      
#training
spec_model = train_CNN_model(mel_specs,epochs=100, batch_size=16, learning_rate=0.0001, title = "Bidirectional model",early_stopping = False, overfit_batch=False, regression = False)


val_size = 0.15

      
#training
reg_spec_model = train_CNN_model(mel_specs,epochs=100, batch_size=16, learning_rate=0.0001, title = "Bidirectional model",early_stopping = False, overfit_batch=False, regression = True)


# In[36]:


from sklearn.metrics import classification_report

def test_CNN_model(model, test_dataset,batch_train):
    #step 7   
    #test_loader = DataLoader(test_dataset, batch_size=batch_train)
    model.eval()
    
    y_predictions,y_real = [], []
    # step 7
    #for test_data in test_loader:
    
    #step 8 
    for test_data in test_dataset:
    
            X_test_batch = test_data[0].float().cuda().view(-1,1,1293,128)
            y_test_batch = test_data[1].cuda()
            lengths = test_data[2].cuda()
            
            out = model(X_test_batch) # feed-forward
            
            # softmax at the last layer
            #only for step 7
            #y_pred = torch.nn.functional.softmax(out, dim = -1).argmax(dim = -1)
            
            #append results 
            #step8
            y_pred = out

            y_pred = y_pred.data.cpu().numpy().tolist()
            y_test_batch = y_test_batch.data.cpu().numpy().tolist()
             
            y_predictions += y_pred
            y_real += y_test_batch
            
    #step 7
    #print(classification_report(y_real, y_predictions))
    #step 8
    #print("Spearman's Correlation is:", scipy.stats.spearmanr(y_real, y_predictions))
    
    ############################## just for step 10 ########################
    for i, name in zip(range(3), ['valence', 'energy', 'danceability']):
        y1 = np.array([a[i] for a in y_real])
        y_pred1 = np.array([a[i] for a in y_predictions])
        print(f"Spearman's Correlation for {name} is: {scipy.stats.spearmanr(y1, y_pred1)}")
    
    return  


batch_train = 14
test_CNN_model(spec_model, mel_specs,batch_train)


batch_train = 14
test_CNN_model(reg_spec_model, mel_specs,batch_train)



mel_specs_overfit = SpectrogramDataset(path='/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',class_mapping=CLASS_MAPPING,train=True,max_length=-1, regression=None, chroma = 'mel')
spec_model_overfit = train_CNN_model(mel_specs_overfit,epochs=30, batch_size=16, learning_rate=0.0001, title = "Bidirectional model",early_stopping = False, overfit_batch=True)


gc.collect()


###################################################
##################### STEP 8 ######################
###################################################


class MultitaskDataset(Dataset):
    def __init__(self, path, max_length=-1, att = 1, chroma = 'mel'):
        t = 'train'
        p = os.path.join(path, t)
        self.chroma = chroma
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        self.files, self.labels = self.get_files_labels(self.index, att)
        self.labels = [float(i) for i in self.labels]
        self.feats = [read_spectrogram(os.path.join(p, f), self.chroma) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        if isinstance(self.labels, (list, tuple)):
            self.labels = np.array(self.labels).astype('float32')


    def get_files_labels(self, txt, att = 1):
        with open(txt, 'r') as fd:
            lines = [l.rstrip().split('\t') for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            label = l[0].split(",")[att]
#             print(label)
            # Kaggle automatically unzips the npy.gz format so this hack is needed
            _id = l[0].split(',')[0]
            npy_file = '{}.fused.full.npy'.format(_id)
            files.append(npy_file)
            labels.append(label)
        return files, labels

    def __getitem__(self, item):
        # TODO: Inspect output and comment on how the output is formatted
        l = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], l

    def __len__(self):
        return len(self.labels)



gc.collect()




multidata_valence = SpectrogramDataset(path = '/kaggle/input/patreco3-multitask-affective-music/data/multitask_dataset',class_mapping=CLASS_MAPPING,train=True,max_length=-1, regression=1, chroma = 'mel')
#vtrain_data, vtest_data = torch_train_val_split(multidata_valenece, batch_size=14 ,batch_eval=14)
valence_regression_lstm_model, test_loader = train_model(multidata_valence, rnn_size=64,num_layers=3,epochs=100,batch_size=10,learning_rate=0.0001, title = "Valence Model",bidirectional = True, dropout=0.2,regularization=0.4,early_stopping = False, overfit_batch=False, regression=True)



test_model(valence_regression_lstm_model, test_loader,batch_train=14)


# In[49]:


multidata_energy = SpectrogramDataset(path = '/kaggle/input/patreco3-multitask-affective-music/data/multitask_dataset',class_mapping=CLASS_MAPPING,train=True,max_length=-1, regression=2, chroma = 'mel')
energy_regression_lstm_model, test_loader = train_model(multidata_energy, rnn_size=64,num_layers=3,epochs=100,batch_size=10,learning_rate=0.0001, title = "Energy Model",bidirectional = True, dropout=0.2,regularization=0.4,early_stopping = False, overfit_batch=False, regression=True)




test_model(energy_regression_lstm_model, test_loader,batch_train=14)



multidata_dance = SpectrogramDataset(path = '/kaggle/input/patreco3-multitask-affective-music/data/multitask_dataset',class_mapping=CLASS_MAPPING,train=True,max_length=-1, regression=3, chroma = 'mel')
danceability_regression_lstm_model, test_loader = train_model(multidata_dance, rnn_size=64,num_layers=3,epochs=100,batch_size=10,learning_rate=0.0001, title = "Danceability_regression_lstm_model Model",bidirectional = True, dropout=0.2,regularization=0.4,early_stopping = False, overfit_batch=False, regression=True)


test_model(danceability_regression_lstm_model, test_loader,batch_train=14)



val_size = 0.15
#mel_specs = SpectrogramDataset(path='/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',class_mapping=CLASS_MAPPING,train=True,max_length=-1, regression=1, chroma = 'mel')
spec_model, test_loader = train_CNN_model(multidata_valence,epochs=100, batch_size=16, learning_rate=0.0001, title = "Bidirectional model",early_stopping = False, overfit_batch=False, regression = True)




test_CNN_model(spec_model, test_loader,14)


gc.collect()


###################################################
##################### STEP 9 ######################
###################################################


cnn_mel = SpectrogramDataset(path='/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',class_mapping=CLASS_MAPPING,train=True,max_length=-1, regression=None, chroma = 'mel')
model_cnn_mel = train_CNN_model(cnn_mel,epochs=100, batch_size=16, learning_rate=0.0001, title = "Bidirectional model",early_stopping = False, overfit_batch=False,regression = False)


# Save the weights
torch.save(model_cnn_mel.state_dict(), 'cnn_mel_mapped_dict')


model_cnn_mel.state_dict().keys()




model_cnn_regression_mel = Cnn(regression = True)
state_dict = torch.load('cnn_mel_mapped_dict')

for key in state_dict:
    if key not in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc2_regression.weight', 'fc2_regression.bias']:
        model_cnn_regression_mel.state_dict()[key] = state_dict[key]



val_size = 0.15
multidata_valence = SpectrogramDataset(path = '/kaggle/input/patreco3-multitask-affective-music/data/multitask_dataset',class_mapping=CLASS_MAPPING,train=True,max_length=-1, regression=1, chroma = 'mel')
spec_model, test_loader = train_CNN_model(model_cnn_regression_mel, multidata_valence, epochs=100, batch_size=16, learning_rate=0.0001, title = "Bidirectional model",early_stopping = False, overfit_batch=False, regression = True)



test_CNN_model(spec_model, test_loader,14)


###################################################
##################### STEP 10 ######################
###################################################


class MultitaskDataset(Dataset):
    def __init__(self, path, max_length=-1, read_spec_fn=read_spectrogram, attribute = 1):
        t = 'train'
        p = os.path.join(path, t)
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        self.files, self.Valence, self.Energy, self.Dance = self.get_files_labels(self.index, attribute)
        self.feats = [read_spec_fn(os.path.join(p, f), 'mel') for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)


    def get_files_labels(self, txt, attribute = 1):
        with open(txt, 'r') as fd:
            lines = [l.rstrip().split('\t') for l in fd.readlines()[1:]]
        files, labels = [], [[],[],[]]
        for l in lines:
            labels1 = l[0].split(",")[1]
            labels2 = l[0].split(",")[2]
            labels3 = l[0].split(",")[3]

            _id = l[0].split(',')[0]
            npy_file = '{}.fused.full.npy'.format(_id)
            files.append(npy_file)
            labels[0].append(float(labels1))
            labels[1].append(float(labels2))
            labels[2].append(float(labels3))
        return files, labels[0], labels[1], labels[2]

    def __getitem__(self, item):
        l = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), torch.tensor([self.Valence[item], self.Energy[item], self.Dance[item]], dtype=torch.double), l

    def __len__(self):
        return len(self.Valence)




class my_loss(torch.nn.Module):
    def __init__(self):
        super(my_loss,self).__init__()
        self.loss = nn.MSELoss()
       
    def forward(self, outputs, labels):
        loss1 = self.loss(outputs[0], labels[0])
        loss2 = self.loss(outputs[1], labels[1])
        loss3 = self.loss(outputs[2], labels[2])
        return loss1 + loss2 + loss3


data = MultitaskDataset(path = '/kaggle/input/patreco3-multitask-affective-music/data/multitask_dataset/', read_spec_fn=read_spectrogram)
model_regression_cnn_mel = Cnn(regression=True)

model_regression_cnn_mel_model, test_loader = train_CNN_model(model_regression_cnn_mel, data,epochs=100, batch_size=16, learning_rate=0.0001, title = "Bidirectional model",early_stopping = False, overfit_batch=False, regression = True)
test_CNN_model(model_regression_cnn_mel_model, test_loader,14)


test_CNN_model(model_regression_cnn_mel_model, test_loader,14)


gc.collect()
