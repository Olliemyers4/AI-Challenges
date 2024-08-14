from jsonschema import validate,ValidationError
from json import loads

def LoadJson(jsonFile):
    with open(jsonFile, 'r') as file:
        data = file.read()
    return loads(data)


def ValidateJson(jsonData, schema):
    validate(instance=jsonData, schema=schema)


schema = LoadJson('plantSchema.json')
plants = LoadJson('plants.json')

for plant in plants:
    ValidateJson(plant, schema)

        

    
