import torch
from pandas import read_csv

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data.astype('float32'))
        self.label = torch.from_numpy(label).long()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

class NeuralClassifier(torch.nn.Module):
    def __init__(self, inputSize, hiddenSize, classes):
        super(NeuralClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(inputSize, hiddenSize)
        self.fc2 = torch.nn.Linear(hiddenSize, classes)
        self.sigmoid = torch.sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

class NeuralTrainer:
    def __init__(self, model, train, valid, test,batchSize,criterion,optimiser, trainingDevice):
        self.model = model.to(trainingDevice)
        self.batchSize = batchSize
        self.criterion = criterion
        self.optimiser = optimiser
        self.trainingDevice = trainingDevice

        self.trainDataSet, self.trainLoader = self.prepData(train)
        self.validDataSet, self.validLoader = self.prepData(valid)
        self.testDataSet, self.testLoader = self.prepData(test)

    def fileToNumpy(self, file):
        df = read_csv(file,header=0)
        return df.to_numpy()

    def prepData(self, data):
        file = self.fileToNumpy(data)
        data, label = file[:,1:], file[:,0]
        dataset = Dataset(data, label).to(self.trainingDevice)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batchSize, shuffle=True).to(self.trainingDevice)  
        return data,loader
    
    def trainModel(self, epochs):
        for i in range(epochs):
            correct,total,loss = 0,0,0
            for j, (data, label) in enumerate(self.trainLoader):
                self.optimiser.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimiser.step()
                loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            validCorrect,validTotal,validLoss = self.validateModel()
        testCorrect,testTotal = self.testModel()
        self.saveModel()

    def validateModel(self):
        correct,total,loss = 0,0,0
        for i, (data, label) in enumerate(self.validLoader):
            output = self.model(data)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimiser.step()
            loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        return (correct,total,loss)
      
    def testModel(self):
        with torch.no_grad():
            correct,total = 0,0
            for i, (data, label) in enumerate(self.testLoader):
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        return (correct,total)

    def saveModel(self):
        torch.save(self.model.state_dict(), 'model.pth')
