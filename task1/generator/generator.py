from jsonschema import validate,ValidationError
from json import loads
import sys
from numpy.random import choice,uniform
from pandas import DataFrame

class Generator:
    def __init__(self, schemaFile, plantFile, trainSize,testSize,validSize,decimalPlaces,preprocess):
        # Load required parameters from command line
        self.schemaFile = schemaFile
        self.schema = self.LoadJson(schemaFile)
        self.plants = self.LoadJson(plantFile)
        self.trainSize = trainSize
        self.testSize = testSize
        self.validSize = validSize
        self.dp = decimalPlaces
        self.preprocess = preprocess

    def LoadJson(self,jsonFile):
        with open(jsonFile, 'r') as file:
            data = file.read()
        return loads(data)

    def ValidateJson(self,jsonData, schema):
        try:
            # Validate the json data against the schema
            # If validation fails, it throws a ValidationError
            validate(instance=jsonData, schema=schema)
        except ValidationError as e:
            print(e)
            sys.exit(1)
        
    def Validate(self):
        for plant in self.plants:
            # Validate each plant against the schema
            self.ValidateJson(plant, self.schema)

    def Generate(self):
        self.GenerateSet(self.trainSize,"train.csv")
        self.GenerateSet(self.testSize,"test.csv")
        self.GenerateSet(self.validSize,"validation.csv")

    def GenerateSet(self,setSize,fileName):
        generatedPlants = []
        for plant in self.plants:
            # Load in static plant attributes
            label = plant["name"]
            climate = plant["climate"]
            shape = plant["shape"]
            flowering = plant["flowering"]
          

            # Load in ranges for height and colour list
            colours = plant["colours"]
            heightmin = plant["heightmin"]
            heightmax = plant["heightmax"]

            # Generate list of attributes in rangeh
            colourList = choice(colours,setSize,replace=True)
            heights = uniform(float(heightmin),float(heightmax),setSize)
            heights = map(lambda x: round(x,self.dp),heights)

            # Generate list of plants
            plantList = map(lambda x,y: {"label":label,"climate":climate,"shape":shape,"flowering":flowering,"colour":x,"height":y},colourList,heights)
            generatedPlants.extend(plantList)
        df = DataFrame(generatedPlants)

        # Preprocessing involves normalising the height values and converting all categorical values to their categorical codes and then normalising them
        if self.preprocess:
            # Normalise height values
            df["height"] = round((df["height"] - df["height"].min())/(df["height"].max()-df["height"].min()),self.dp)

            # Normalise categorical values
            for col in ["climate","shape","flowering","colour"]: # Categorical columns
                df[col] = df[col].astype('category').cat.codes
                df[col] = round((df[col] - df[col].min())/(df[col].max()-df[col].min()),self.dp)
        
            # Convert label to categorical codes - don't normalise this
            df["label"] = df["label"].astype('category').cat.codes
            
        df.to_csv(fileName,index=False)
        



if __name__ == "__main__":
    args = sys.argv
    if len(args) != 8: # Enforce correct usage
        print("Usage: python generator.py plantSchema.json plants.json TrainSize TestSize ValidationSize DecimalPlacesOnHeight,preprocess(0/1)")
        sys.exit(1)
    g = Generator(args[1], args[2], int(args[3]), int(args[4]), int(args[5]), int(args[6]), bool(args[7]))
    g.Validate()
    g.Generate()


        

    
