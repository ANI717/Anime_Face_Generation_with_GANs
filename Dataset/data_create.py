import os
import pandas as pd

data = os.listdir('images')
pd.DataFrame(data ,columns=["images"]).to_csv('data.csv', index=False)