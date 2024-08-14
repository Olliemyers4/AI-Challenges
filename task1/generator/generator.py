from jsonschema import validate,ValidationError
from json import loads
import sys
from numpy.random import choice,uniform
from pandas import DataFrame

class Generator:
    def __init__(self, schemaFile, plantFile, perPlant,decimalPlaces):
        self.schemaFile = schemaFile
        self.plantFile = plantFile
        self.perPlant = perPlant
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
        generatedPlants = []
        for plant in self.plants:
            label = plant["name"]
            climate = plant["climate"]
            shape = plant["shape"]
            flowering = plant["flowering"]
            colours = plant["colours"]
            heightmin = plant["heightmin"]
            heightmax = plant["heightmax"]
            colourList = choice(colours,self.perPlant,replace=True)
            heights = uniform(float(heightmin),float(heightmax),self.perPlant)
            heights = map(lambda x: round(x,self.dp),heights)
            plantList = map(lambda x,y: {"label":label,"climate":climate,"shape":shape,"flowering":flowering,"colour":x,"height":y},colourList,heights)
            generatedPlants.extend(plantList)
        df = DataFrame(generatedPlants)
        df.to_csv("generatedPlants.csv",index=False)
        



if __name__ == "__main__":
    args = sys.argv
    if len(args) != 5:
        print("Usage: python generator.py plantSchema.json plants.json SamplesPerPlant DecimalPlacesOnHeight")
        sys.exit(1)
    g = Generator(args[1], args[2], int(args[3]), int(args[4]))
    g.Validate()
    g.Generate()


        

    
