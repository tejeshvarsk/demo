
import os
import joblib
from joblib import load
import pandas as pd
import io

def load_model():
    model_directory = os.path.dirname(os.path.realpath(__file__))
    model_filename = "model.joblib"
    contents = os.listdir(model_directory)
    if model_filename in contents:
        with open(os.path.join(model_directory, model_filename), "rb") as file:
            model = load(file)
    else:
        raise Exception("{0} file not present in {1} directory".format(model_filename, model_directory))
    return model

def predict(data, model=load_model()):
    assert model is not None, "model not loaded"
    X =  pd.read_json(io.StringIO(data) ) if isinstance(data, str) else pd.DataFrame.from_dict(data)
    preds = model.predict(X).tolist()
    return {"predictions": preds}
