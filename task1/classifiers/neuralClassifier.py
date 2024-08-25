import torch
from pandas import read_csv
import matplotlib.pyplot as plt
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data.astype('float32'))
        self.label = torch.from_numpy(label).long()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

class NeuralClassifier(torch.nn.Module):
    def __init__(self, inputSize, hiddenSize, classes,activation):
        super(NeuralClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(inputSize, hiddenSize)
        self.fc2 = torch.nn.Linear(hiddenSize, classes)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class NeuralTrainer:
    def __init__(self, model, train, valid, test,batchSize,criterion,optimiser,name, trainingDevice):
        self.model = model.to(trainingDevice)
        self.batchSize = batchSize
        self.criterion = criterion
        self.optimiser = optimiser
        self.trainingDevice = trainingDevice
        self.name = name

        self.trainDataSet, self.trainLoader = self.prepData(train)
        self.validDataSet, self.validLoader = self.prepData(valid)
        self.testDataSet, self.testLoader = self.prepData(test)

        plt.ion()
        self.trainLoss = []
        self.validLoss = []


    def fileToNumpy(self, file):
        df = read_csv(file,header=0)
        return df.to_numpy()

    def prepData(self, data):
        file = self.fileToNumpy(data)
        data, label = file[:,1:], file[:,0]
        dataset = Dataset(data, label)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batchSize, shuffle=True)
        return data,loader
    
    def trainModel(self, epochs):
        for i in range(epochs):
            correct,total,loss = 0,0,0
            for j, (data, label) in enumerate(self.trainLoader):
                data = data.to(self.trainingDevice)
                label = label.to(self.trainingDevice)

                self.optimiser.zero_grad()
                output = self.model(data)
                losses = self.criterion(output, label)
                losses.backward()
                self.optimiser.step()
                loss += losses.item() 

                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            validCorrect,validTotal,validLoss = self.validateModel()
            self.trainLoss.append(loss/len(self.trainLoader))
            self.validLoss.append(validLoss/len(self.validLoader)) # Normalise loss
            self.plotGraph()
        testCorrect,testTotal = self.testModel()
        print(f'Test Set Accuracy: {correct/total*100}%')
        self.plotGraph()
        plt.ioff()
        plt.show()
     

        self.saveModel()

    def validateModel(self):
        self.model.eval()
        correct,total,loss = 0,0,0
        with torch.no_grad():
            for i, (data, label) in enumerate(self.validLoader):
                data = data.to(self.trainingDevice)
                label = label.to(self.trainingDevice)

                output = self.model(data)
                losses = self.criterion(output, label)
                loss += losses.item()
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        self.model.train()
        return (correct,total,loss)
      
    def testModel(self):
        self.model.eval()
        with torch.no_grad():
            correct,total = 0,0
            for i, (data, label) in enumerate(self.testLoader):
                data = data.to(self.trainingDevice)
                label = label.to(self.trainingDevice)

                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        return (correct,total)

    def saveModel(self):
        torch.save(self.model.state_dict(), f'{self.name}.pth')

    def plotGraph(self):
        plt.figure(1)
        plt.clf()
        plt.plot(self.trainLoss, label='Training Loss')
        plt.plot(self.validLoss, label='Validation Loss')
        plt.legend()
        ax = plt.gca()
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.pause(0.1)
        pass
