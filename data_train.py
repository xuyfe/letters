# the original data is too large, we sample 50%.

import pandas as pd
import numpy as np

df = pd.read_csv('data/emnist-letters-train.csv')
df = df.sample(frac=0.5, replace=False)

df.to_csv('data/train_data.csv', index=False)