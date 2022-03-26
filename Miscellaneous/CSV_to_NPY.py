import pandas as pd
import numpy as np

df = pd.read_csv('../decals/Decals_Segment_Votes_61563.csv')

X =np.array(df.iloc[0:61564, 1:35])

print(X)
print(X.shape)

np.save('../decals/Decals_Segmented_Votes_61563.npy', X)

#print(X[60891])