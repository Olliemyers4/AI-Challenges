from jsonschema import validate,ValidationError
from json import loads
import sys
from numpy.random import choice,uniform
from pandas import DataFrame

class Generator:
    def __init__(self, schemaFile, plantFile, trainSize,testSize,validSize,decimalPlaces):
        self.schemaFile = schemaFile
        self.plantFile = plantFile
        self.trainSize = trainSize
        self.testSize = testSize
        self.validSize = validSize
        self.dp = decimalPlaces
        self.schema = None
        self.plants = None

    def LoadJson(self,jsonFile):
        with open(jsonFile, 'r') as file:
            data = file.read()
        return loads(data)

    def ValidateJson(self,jsonData, schema):
        try:
            validate(instance=jsonData, schema=schema)
        except ValidationError as e:
            print(e)
            sys.exit(1)
        
    def Validate(self):
        self.schema = self.LoadJson(self.schemaFile)
        self.plants = self.LoadJson(self.plantFile)

        for plant in self.plants:
            self.ValidateJson(plant, self.schema)

    def Generate(self):
        self.GenerateSet(self.trainSize,"train.csv")
        self.GenerateSet(self.testSize,"test.csv")
        self.GenerateSet(self.validSize,"validation.csv")

    def GenerateSet(self,setSize,fileName):
        generatedPlants = []
        for plant in self.plants:
            label = plant["name"]
            climate = plant["climate"]
            shape = plant["shape"]
            flowering = plant["flowering"]
            colours = plant["colours"]
            heightmin = plant["heightmin"]
            heightmax = plant["heightmax"]
            colourList = choice(colours,setSize,replace=True)
            heights = uniform(float(heightmin),float(heightmax),setSize)
            heights = map(lambda x: round(x,self.dp),heights)
            plantList = map(lambda x,y: {"label":label,"climate":climate,"shape":shape,"flowering":flowering,"colour":x,"height":y},colourList,heights)
            generatedPlants.extend(plantList)
        df = DataFrame(generatedPlants)
        df.to_csv(fileName,index=False)
        



if __name__ == "__main__":
    args = sys.argv
    if len(args) != 7:
        print("Usage: python generator.py plantSchema.json plants.json TrainSize TestSize ValidationSize DecimalPlacesOnHeight")
        sys.exit(1)
    g = Generator(args[1], args[2], int(args[3]), int(args[4]), int(args[5]), int(args[6]))
    g.Validate()
    g.Generate()


        

    
