import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv('../Data/SersicVotes.csv')


#Load GZ votes and CapsNet predictions
x = df["Class11"].to_numpy()
y = df["RGB0"].to_numpy()
colors = df["ng"].to_numpy()


#Plot scatter diagram
plt.scatter(x,y,s=10, c=colors, cmap=plt.cm.get_cmap("jet"))

#Create Colour bar
cbar = plt.colorbar(orientation="vertical",
                   pad=0.05, shrink=1, aspect=20, format="%.0f")
cbar.set_label(label="Sersic Index", fontsize=13)
cbar.set_ticks([0,5,10])
cbar.ax.tick_params(labelsize=15)

#Format graph
plt.title("CapsNet smooth predictions Vs Galaxy Zoo smooth votes", fontsize=18)
plt.xlabel("Galaxy Zoo Votes", fontsize=13)
plt.ylabel("CapsNet Predictions", fontsize=13)
plt.clim(0,10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim([0, 1])
plt.xlim([0, 1])
plt.plot([0, 1], [0, 1], color="black", linestyle="-.", label= 'Matching Output')
plt.show()