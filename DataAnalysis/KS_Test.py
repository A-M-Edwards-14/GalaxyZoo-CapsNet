from scipy import stats
import numpy as np
import pandas as pd


# df = pd.read_csv('../SersicTable/BlueSmooth.csv')
# df2 = pd.read_csv('../SersicTable/CapsNetBlueSmooth.csv')
df = pd.read_csv('../SersicTable/RedSmooth.csv')
df2 = pd.read_csv('../SersicTable/CapsNetRedSmooth.csv')


# df = pd.read_csv('../SersicTable/BlueSpirals.csv')
# df2 = pd.read_csv('../SersicTable/CapsNetBlueSpiral.csv')
#df = pd.read_csv('../SersicTable/RedSpirals.csv')
#df2 = pd.read_csv('../SersicTable/CapsNetRedSpiral.csv')

#Extract sersic index from csv files
x= df['ng'].to_numpy()
y= df2['ng'].to_numpy()


print(stats.kstest(x, y))

print(stats.anderson(x, dist='norm'))

print(stats.anderson(y, dist='norm'))
# print(stats.kstest(stats.norm.rvs(size=100, random_state=rng), stats.norm.cdf))
# print(stats.kstest(stats.norm.rvs, 'norm', N=100))
