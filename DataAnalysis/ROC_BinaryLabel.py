import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


df = pd.read_csv('../GreenValleyData/GalaxyROCData.csv')

#Load binary ground truth labels, these have been rounded to 1 or 0 from GZ votes
y_truee = df["0.5Class1.1"]
y_truee2 = df["0.5Class1.2"]

y_true = df["0.8Class1.1"]
y_true2 = df["0.8Class1.2"]

#Load fractional CapsNet predictions
y_pred = df["RGB0"]
y_pred2 = df["RGB1"]

y_predGrey = df["Grey0"]
y_predGrey2 = df["Grey1"]



# Function to calculate True Positive Rate and False Positive Rate

def calc_TP_FP_rate(Z_true, Z_pred):
    
    # Convert predictions to series with index matching y_true
    Z_pred = pd.Series(Z_pred, index=Z_true.index)
    
    # Instantiate counters
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # Determine whether each prediction is TP, FP, TN, or FN
    for i in Z_true.index: 
        if Z_true[i]==Z_pred[i]==1:
           TP += 1
        if Z_pred[i]==1 and Z_true[i]!=Z_pred[i]:
           FP += 1
        if Z_true[i]==Z_pred[i]==0:
           TN += 1
        if Z_pred[i]==0 and Z_true[i]!=Z_pred[i]:
           FN += 1
    
    # Calculate true positive rate and false positive rate
    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)

    return tpr, fpr


# Containers for true positive / false positive rates
def multi(y_pred, y_true):
    lr_tp_rates = []
    lr_fp_rates = []

    # Define probability thresholds to use, between 0 and 1
    probability_thresholds = np.linspace(0,1,num=100)

    # Find true positive / false positive rate for each threshold
    for p in probability_thresholds:
        print(p)
        ##np.around(y_pred-p)
        y_test_preds = []
        
        for prob in y_pred:
            if prob > p:
                y_test_preds.append(1)
            else:
                y_test_preds.append(0)
                
        tp_rate, fp_rate = calc_TP_FP_rate(y_true, y_test_preds)
            
        lr_tp_rates.append(tp_rate)
        lr_fp_rates.append(fp_rate)
    return lr_fp_rates, lr_tp_rates

#print(auc(lr_fp_rates, lr_tp_rates))
###############

x,y = multi(y_pred, y_true)
x2,y2 = multi(y_pred2, y_true2)
xGrey,yGrey = multi(y_predGrey, y_true)
xGrey2,yGrey2 = multi(y_predGrey2, y_true2)

aucRGB = auc(x,y)
aucRGB2 = auc(x2,y2)
aucGrey = auc(xGrey,yGrey)
aucGrey2 = auc(xGrey2,yGrey2)



plt.plot(x,y, color='red', linestyle='-', label= f'RGB Smooth, AUC: {np.round(aucRGB,3)}')
plt.plot(x2,y2, color='blue', linestyle='-', label= f'RGB Featured, AUC: {np.round(aucRGB2,3)}')
plt.plot(xGrey,yGrey, color='grey', linestyle='--', label= f'Grey Smooth, AUC: {np.round(aucGrey,3)}')
plt.plot(xGrey2,yGrey2, color='black', linestyle='--', label= f'Grey Featured, AUC: {np.round(aucGrey2,3)}')
plt.plot([0, 1], [0, 1], color="orange", linestyle="-.", label= 'Untrained')

plt.legend()
plt.title("CapsNet ROC curve using a ground truth threshold of 0.8")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

#plt.plot(lr_fp_rates,lr_tp_rates)
plt.show()





xhalf,yhalf = multi(y_pred, y_truee)
x2half,y2half = multi(y_pred2, y_truee2)
xGreyhalf,yGreyhalf = multi(y_predGrey, y_truee)
xGrey2half,yGrey2half = multi(y_predGrey2, y_truee2)

aucRGBhalf = auc(xhalf,yhalf)
aucRGB2half = auc(x2half,y2half)
aucGreyhalf = auc(xGreyhalf,yGreyhalf)
aucGrey2half = auc(xGrey2half,yGrey2half)



plt.plot(xhalf,yhalf, color='red', linestyle='-', label= f'RGB Smooth, AUC: {np.round(aucRGBhalf,3)}')
plt.plot(x2half,y2half, color='blue', linestyle='-', label= f'RGB Featured, AUC: {np.round(aucRGB2half,3)}')
plt.plot(xGreyhalf,yGreyhalf, color='grey', linestyle='--', label= f'Grey Smooth, AUC: {np.round(aucGreyhalf,3)}')
plt.plot(xGrey2half,yGrey2half, color='black', linestyle='--', label= f'Grey Featured, AUC: {np.round(aucGrey2half,3)}')
plt.plot([0, 1], [0, 1], color="orange", linestyle="-.", label= 'Untrained')

plt.legend()
plt.title("CapsNet ROC curve using a ground truth threshold of 0.5")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

#plt.plot(lr_fp_rates,lr_tp_rates)
plt.show()


