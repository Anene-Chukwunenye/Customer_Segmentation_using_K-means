import pandas as pd
import numpy as np
from fastapi import FastAPI
import pickle
from models import Customer



with open("./model.pkl", "rb") as f:
    model = pickle.load(f)
app = FastAPI()

with open("./scale.pkl", "rb") as f:
     scaler = pickle.load(f)

@app.get("/")
def greetings():
    return "Hello, What are you doing here? Get to the docs and get useful information.."

@app.post("/predict_cluster")
def cluster(customer:Customer):
        customer = customer.dict()
        df = pd.DataFrame(customer, index=[0])

        dfo = pd.read_csv("./data.csv")
        df['housing']=df['housing'].map({'no':0,'yes':1})
        df['loan']=df['loan'].map({'no':0,'yes':1})
        df['subscribed']=df['subscribed'].map({'no':0,'yes':1})

        processedDf = pd.get_dummies(df, columns=['job','education','marital','default','poutcome'], dtype=int)
        missing_cols = set(dfo.columns) - set(processedDf.columns)
        for col in missing_cols:
            processedDf[col] = 0
        processedDf = processedDf.reindex(columns=dfo.columns[:], fill_value=0)

        prescaledDf = scaler.transform(processedDf)
        scaledDf = pd.DataFrame(prescaledDf, columns = processedDf.columns)

        return f"This example belongs to cluster - {str(model.predict(scaledDf))}"


#Ignore. For the sake of Debugging
"""print(cluster({
  "age": 30,
  "job": "tier1",
  "marital": "married",
  "education": "non-educated",
  "default": "no",
  "housing": "no",
  "loan": "no",
  "campaign": 1,
  "previous": 1,
  "poutcome": "failure",
  "subscribed": "no"
}))


"""