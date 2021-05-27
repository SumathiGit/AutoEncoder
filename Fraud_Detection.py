"""
Using a Autoencoder we are importing a dataset(source:kaggle.com) which contains the credit card transaction with fraud and non- Fraud transaction (class 0's >> non_fraud , 1's >> Fraud)
Auto_Encoders are nothing but a Unsupervised Learning method to learn the most important features from the dataset an a input  in a low dimensional space and then reconstruct the output from the input  
we are not gonna use any labels for the target,just prdicting how the output similar to the input using encoding and decoding
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from torch.autograd import Variable
from torch.utils.data import DataLoader


from sklearn.preprocessing import StandardScaler            #We are using sklearn to normalize our datas
from sklearn.model_selection import train_test_split



data = pd.read_csv("/home/sumathi/Spyder 3/archive/creditcard.csv")
print(data) #After visualizing our dataset there are 284807 rows of credit card transacton has been recorded as well as Time , Amount, Class which are not normalized ,>> only v1,v2,..,v28 values are normalized
fraud_case = data[data['Class'] == 1]
ok_case = data[data['Class'] == 0]

plt.plot(data['Time'], data['Amount'])

groups = data.groupby('Class')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.Time, group.Amount, marker='o', linestyle='', ms=5, label=name)
    ax.legend()
    plt.show()
    
    
data['Time'] = data['Time'].apply(lambda x: np.ceil(float(x)/3600)% 24.)                      #Changing the time from seconds to hour
scl =StandardScaler()
data['Time'] = scl.fit_transform(data['Time'].values.reshape(-1, 1))
scl = StandardScaler()
data['Amount'] = scl.fit_transform(data['Amount'].values.reshape(-1, 1))
#Here wew are spliting our dataset into training dataset and testing dataset out of 100% of dataset we will give 0.2 >> 20% of data for the testing dataset
x_train, x_test = train_test_split(data, test_size = 0.2, random_state = 42)
x_train.shape               #(227845, 31)
x_test.shape                #(56962, 31)
#Here we are gonna reduce the class from both training dataset and test dataset  to remove the fraud_transaction and store the values in y_test(0's and 1's) 
x_train = x_train[x_train['Class'] == 0]
x_train.shape

x_train = x_train.drop("Class", axis = 1)
x_train.shape

y_test = x_test['Class'].values
y_test.shape              

x_test = x_test.drop("Class", axis = 1)
x_test = x_test.values
x_train = x_train.values

x_train.shape, x_test.shape  #((227451, 30)<== non-fraud data(For training purpose), (56962, 30)<== all data(For test))

#Next changing the datasets into tensor. after that, Because of huge dataset we are using dataloader with the specific batch size
train_tensor = torch.FloatTensor(x_train)
test_tensor = torch.FloatTensor(x_test)
train_loader = DataLoader(train_tensor, batch_size = 1000)
test_loader = DataLoader(test_tensor, batch_size = 1000)
 
class AutoEncoder_Fraud(nn.Module):
    def __init__(self):
        super(AutoEncoder_Fraud, self).__init__()
        self.fc1 = nn.Linear(30, 20)      #First layer with 30 neurons and 20 and then 20 to 10 (fc1 is our convenient variable)
        self.fc2 = nn.Linear(20, 10)      #Encoding  >> Encoder : - a function f that compresses the input into a latent-space representation. f(x) = h where x is a orig data
        self.fc3 = nn.Linear(10, 20)      #Decoding  >> Decoder : - a function g that reconstruct the input from the latent space representation. g(h) ~ x
        self.fc4 = nn.Linear(20, 30)
        
        self.activation = nn.Sigmoid()     #we have to predict the probability as an output. Since probability of anything exists only between the range of 0 and 1 
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        
        return (x)
    
model = AutoEncoder_Fraud()
loss = nn.MSELoss()    # Mean squared error is calculated as the average of the squared differences between the predicted and actual values. 
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
        
for epoch in range(30):
    losses = []
    train_data_loader = iter(train_loader)      #Train_loader will load the train tensor of non_fraud data for training
    for t in range (len(train_data_loader)):
        data = next(train_data_loader)
        data_v = Variable(data)
        y_pred = model(data_v)
        l = loss(y_pred, data_v)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print("Epoch : {%s} loss : {%s}" % (epoch, l.data))
    
    
#From the epoch the loss gradually decresing so now we are gonna test the sample with x_test which contains all the data
    
test_data_loader =iter(test_loader) #Test_loader which contains the test tensors
preds = []
for t in range(len(test_data_loader)):
    data = next(test_data_loader)
    data_v = Variable(data)
    y_pred = model(data_v)
    print("Loss ->", loss(y_pred, data_v).data)


#From the loss we can assume that the loss more than 0.19 and 0.20 are fraud _case transactions
#Intead of manually checcking the fraud transaction wwe can compare the loss with original dataset we are storing the value in y_test --> 0's and 1's for fraud and non_fraud transaction
 
 
 """
  Applications of Autoencoders

    Dimensionality Reduction
    Image Compression
    Image Denoising
    Feature Extraction
    Image generation
    Sequence to sequence prediction
    Recommendation system
"""