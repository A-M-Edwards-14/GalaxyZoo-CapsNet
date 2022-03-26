import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#CapsNet RGB predictions
csv = pd.read_csv('/content/drive/MyDrive/IndexedKaggle_Schawinski.csv')

#CapsNet Greyscale predictions
#csv = pd.read_csv('/content/drive/MyDrive/IndexedKaggle_Schawinski_Grey.csv')

#ResNet152RGB Predicitions
#csv = pd.read_csv('/content/drive/MyDrive/GreenValleyResNet_Final.csv')

#ResNet152 Greyscale Predicitions
#csv = pd.read_csv('/content/drive/MyDrive/GreenValleyPred37GreyFINAL.csv')

#selecting rows based on condition, early type condition
rslt_df = csv[csv['Class1.1'] > 0.8]
#rslt_df = csv[csv['0'] > 0.8]

#Late type spiral condition
rslt_df2 = csv[csv['Class1.2'] > 0.8]
#rslt_df2 = csv[csv['1'] > 0.8]


fig, ax = plt.subplots()
# Basic 2D density plot
# sns.kdeplot(data=csv, x="LOG_MSTELLAR", y="U-R", cmap ='Greens')
#sns.kdeplot(data=csv, x="LOG_MSTELLAR", y="U-R")

#Add in the two linear green valley lines.
x1 = np.linspace(9, 12)
y1 = 0.25*x1 - 0.24
sns.lineplot(x=x1, y=y1, ax=ax, color='green')

x2 = np.linspace(9, 12)
y2 = 0.25*x1 - 0.75
sns.lineplot(x=x2, y=y2, ax=ax, color='green')

#Create the Gaussian KDE plot using seaborn

#Early-type plot
sns.kdeplot(data=rslt_df, x="LOG_MSTELLAR", y="U_R",levels=6, thresh=.2, color='black', ax=ax)
#Late-type plots
#sns.kdeplot(data=rslt_df2, x="LOG_MSTELLAR", y="U_R",levels=6, thresh=.2, color='black', ax=ax)
#All galaxies plot
#sns.kdeplot(data=csv, x="LOG_MSTELLAR", y="U_R",levels=6, thresh=.2, color='black', ax=ax)


#Format the plot

#plt.title("A Colour-Mass plot of all Galaxy types in a sample")
#plt.title("A CapsNet predicted Colour-Mass plot of early-type \n galaxies, using RGB images", fontsize=16)
#plt.title("A ResNet predicted Colour-Mass plot of late-type \n galaxies, using greyscale images", fontsize=16)
plt.title("A Colour-Mass plot of early-type galaxies", fontsize=18)
plt.xlabel("Stellar Mass (Log($M_{*}$/$M_{\odot}$))", fontsize=15)
plt.ylabel("u-r colour (dust corrected)", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
ax.set_ylim(0.8, 3.25)
ax.set_xlim(8.75, 12.25)
plt.show()