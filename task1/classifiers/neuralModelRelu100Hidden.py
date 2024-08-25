import neuralClassifier as nc
import pandas as pd
import torch

#Load data
train = 'train.csv'
valid = 'validation.csv'
test = 'test.csv'


trainData = pd.read_csv(train)

#Initialise model
inputParams = trainData.shape[1] - 1 #Number of columns in the dataset minus the label column
hiddenParams = 100
outputParams = trainData['label'].nunique()
activation = torch.nn.ReLU()
model = nc.NeuralClassifier(inputParams, hiddenParams, outputParams, activation)

#Initialise trainer
batchSize = 100
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
name = 'neuralModelRelu100Hidden'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainer = nc.NeuralTrainer(model, train, valid, test, batchSize, criterion, optimizer,name, device)

trainer.trainModel(1000)
