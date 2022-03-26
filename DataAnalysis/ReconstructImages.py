import numpy as np
import matplotlib.pyplot as plt

#X=np.load(r"C:\Users\USER\Documents\____MastersProject\ReconstructionResults\confs.npy")
Y=np.load('../ReconstructionResults/ground_truth_200_epoch.npy')
Z=np.load('../ReconstructionResults/reconstruction_200_epoch.npy')


# Original images
# for i, el in enumerate(Y):
#     #moving axis to use plt: i.e [4,100,100] to [100,100,4]
#     array2 = np.moveaxis(Y[i], 0, -1)
#     plt.subplot(4, 5, i + 1)
#     plt.imshow(array2, cmap ='gray', interpolation='nearest', ) 
#     plt.axis('off')

# plt.suptitle('Original Images')  
# plt.show()

#Reconstructed images
for i, el in enumerate(Z):
    #moving axis to use plt: i.e [4,100,100] to [100,100,4]
    array3 = np.moveaxis(Z[i], 0, -1)
    plt.subplot(4, 5, i + 1)
    plt.imshow(array3, cmap ='gray') 
    plt.axis('off')
    #plt.subplots_adjust(wspace=0, hspace=0) 

plt.suptitle('Reconstruction after 200 epochs')    
plt.show()
