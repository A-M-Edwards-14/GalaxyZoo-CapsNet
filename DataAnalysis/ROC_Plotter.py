import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score

#Load csv which contains both the Galaxy Zoo vote fractions and Network predictions

# df = pd.read_csv('../GreenValleyData/KaggleROCData.csv')
df = pd.read_csv('../ResNet/ResNetGreenValley/GreenValleyResNet_Final.csv')

#Load Galaxy Zoo votes
y_true = df["Class1.1"]
y_true2 = df["Class1.2"]

#Load Predictions
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
    if TP+FN == 0:
         tpr = 0
         fpr = 0
    else:
        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)

    return tpr, fpr


# Containers for true positive / false positive rates
def multi(Y_pred, Y_true):
    lr_tp_rates = []
    lr_fp_rates = []

    # Define probability thresholds to use, between 0 and 1
    probability_thresholds = np.linspace(0,1,num=100)

    # Find true positive / false positive rate for each threshold
    for p in probability_thresholds:
        print(p)
        y_test_preds = []
        y_trueGZ = []
        
        for prob in Y_pred:
            if prob > p:
                y_test_preds.append(1)
            else:
                y_test_preds.append(0)
        
        for prob in Y_true:
            if prob > p:
                y_trueGZ.append(1)
            else:
                y_trueGZ.append(0)
                
        y_trueGZ = pd.Series(y_trueGZ)
        tp_rate, fp_rate = calc_TP_FP_rate(y_trueGZ, y_test_preds)
        print(tp_rate, fp_rate)
            
        lr_tp_rates.append(tp_rate)
        lr_fp_rates.append(fp_rate)
    return lr_fp_rates, lr_tp_rates



x,y = multi(y_pred, y_true)
x2,y2 = multi(y_pred2, y_true2)
xGrey,yGrey = multi(y_predGrey, y_true)
xGrey2,yGrey2 = multi(y_predGrey2, y_true2)

aucRGB = auc(x,y)
aucRGB2 = roc_auc_score(x2,y2)
aucGrey = auc(xGrey,yGrey)
aucGrey2 = auc(xGrey2,yGrey2)



plt.plot(x,y, color='red', linestyle='-', label= f'RGB Smooth, AUC: {np.round(aucRGB,3)}')
plt.plot(x2,y2, color='blue', linestyle='-', label= f'RGB Featured, AUC: {np.round(aucRGB2,3)}')
plt.plot(xGrey,yGrey, color='grey', linestyle='--', label= f'Grey Smooth, AUC: {np.round(aucGrey,3)}')
plt.plot(xGrey2,yGrey2, color='black', linestyle='--', label= f'Grey Featured, AUC: {np.round(aucGrey2,3)}')
plt.plot([0, 1], [0, 1], color="orange", linestyle="-.", label= 'Untrained')

plt.legend()
# plt.title("ROC curve of CapsNet predictions Vs Galaxy Zoo vote fractions")
plt.title("ROC curve of ResNet predictions Vs Galaxy Zoo vote fractions")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

#plt.plot(lr_fp_rates,lr_tp_rates)
plt.show()