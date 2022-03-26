import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load the histogram data
df = pd.read_csv('../SersicTable/BlueSmooth.csv')
df2 = pd.read_csv('../SersicTable/RedSmooth.csv')

#df = pd.read_csv('../SersicTable/BlueSpiral.csv')
#df2 = pd.read_csv('../SersicTable/RedSpiral.csv')


#Define the number of bins
binned=[]
x=0
for i in range(0,31):
    binned.append(x)
    x+=0.25
#print(bins)



(n, bins, patches) = plt.hist(df['ng'], bins=binned, histtype= 'step', fill =None, color ='blue', label='Blue Ellipticals')
(n2, bins2, patches2) = plt.hist(df2['ng'], bins=binned, histtype= 'step', fill =None, color='red', label='Red Ellipticals')
#(n3, bins3, patches2) = plt.hist(df3['SmoothVote'], bins=binned, histtype= 'step', fill =None, color='green')
#bins=40

#bin the counts in each bin
#print(n)
print(n2)
#print(n3)

#Format plot
#plt.title("The distribution of red and blue featured galaxies \n by Sersic index, predicted by CapsNet" , fontsize=18)
#plt.title("The distribution of red and blue smooth galaxies \n by Sersic index, predicted by CapsNet" , fontsize=18)
#plt.title("The distribution of red and blue featured galaxies \n by Sersic index" , fontsize=18)
plt.title("The distribution of red and blue smooth galaxies \n by Sersic index", fontsize=18)
plt.xlabel("Sersic Index", fontsize=13)
plt.ylabel("Number of Galaxies", fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(prop={'size': 15}, loc = 'upper left')

plt.show()