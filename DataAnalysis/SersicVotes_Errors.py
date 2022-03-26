from cgitb import grey
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats

#df = pd.read_csv('../GreenValleyData/SersicTable/SersicGreenValley.csv')
df = pd.read_csv('../GreenValleyData/SersicTable/RedGalaxies.csv')
#df = pd.read_csv('../GreenValleyData/SersicTable/BlueGalaxies.csv')
#df = pd.read_csv('../GreenValleyData/SersicTable/GreenGalaxies.csv')

RGB0 = df["ActualMinusPredRGB0"].to_numpy()
RGB1 = df["ActualMinusPredRGB1"].to_numpy()
Grey0 = df["ActualMinusPredGrey0"].to_numpy()
Grey1 = df["ActualMinusPredGrey1"].to_numpy()
Sersic = df["ng"].to_numpy()

def percentile10(y):
   return(np.percentile(y,10))

binned=[]
x=0.6
for i in range(0,15):
    binned.append(x)
    x+=0.6


#X:Statistic, Y:bin edges, Z: Bin Number.
X, Y, Z = scipy.stats.binned_statistic(Sersic, RGB0, statistic='median', bins=15, range=(0,10))
X16, Y16, Z16 = scipy.stats.binned_statistic(Sersic, RGB0, statistic=lambda y: np.percentile(y, 16), bins=15, range=(0,10))
X84, Y84, Z84 = scipy.stats.binned_statistic(Sersic, RGB0, statistic=lambda y: np.percentile(y, 84), bins=15, range=(0,10))

print(X)
print(X16)
print(X84)

X16_error = X16-X
X84_error = X-X84

plt.scatter(Sersic, RGB0, alpha =0.1, color="red")
plt.errorbar(binned, X, yerr=[X16_error, X84_error], color="black", ls="none", capsize=2, label="Central 68% of distribution")
plt.scatter(binned, X, color="black", label="Median")
#plt.title("The difference between Galaxy Zoo votes and CapsNet \n predictions against Sersic Index for green galaxies", fontsize=15)
#plt.title("The difference between Galaxy Zoo votes and CapsNet \n predictions against Sersic Index for blue galaxies", fontsize=15)
plt.title("The difference between Galaxy Zoo votes and CapsNet \n predictions against Sersic Index for red galaxies", fontsize=15)
#plt.title("The difference between Galaxy Zoo votes and CapsNet \n predictions against Sersic Index", fontsize=15)
plt.xlabel("Sersic Index", fontsize=13)
plt.ylabel("GZ vote - CapsNet prediction", fontsize=13)
plt.xlim([0, 9])
plt.ylim([-0.75, 0.75])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(prop={'size': 13}, loc = 'upper right')
plt.axhline(y=0, color="black", linestyle="--")
plt.show()
