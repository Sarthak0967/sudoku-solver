import pickle
import yaml

def load_config():
    
    with open("config.yaml") as p:
        config = yaml.safe_load(p)
    return config

def pickle_dump(path, variable):
    with open(path, "wb") as handle:
        pickle.dump(variable, handle)
        
def pickle_load(path):
    with open(path, "rb") as handle:
        loaded = pickle.load(handle)
    return loaded