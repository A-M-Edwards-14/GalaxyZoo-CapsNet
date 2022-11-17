# ImageFilter for using filter() function
from PIL import Image, ImageFilter
from skimage import io
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from skimage import data, img_as_float
from torchvision import transforms 
import numpy as np
import matplotlib.pyplot as plt

#Images: \J010\J010002.98-045610.5.png , J011\J011001.20+002441.9.png , J011\J011002.57+050018.4.png , J011\J011002.82+002515.1.png
#J014010.30-103952.7, \J014000.75-083628.6.png <- USE THIS
# image = Image.open(r"C:\Users\USER\Documents\____MastersProject\decals\TestData\images\J014000.75-083628.6.png")
image = Image.open(r"C:\Users\Anton (Main)\Desktop\!extracted\sampleGalaxies\100474.jpg")

# transform=transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop((216,216)), transforms.Resize((72,72)), transforms.ToTensor()])
transform=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((216,216)), transforms.Resize((72,72)), transforms.Grayscale(num_output_channels=1), transforms.ToPILImage()])
transformRGB=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((216,216)), transforms.Resize((72,72)), transforms.ToPILImage()])
transformTensor=transforms.Compose([transforms.ToTensor()])
Transformed = transform(image)

transform1 = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((216,216)), transforms.ToPILImage()])

org_img= transform1(image)
array = np.reshape(org_img, (216, 216, 3))

#Display original image
#plt.imshow(array, cmap=plt.cm.gray)
plt.tight_layout()
plt.imshow(array)
plt.axis('off')
plt.tight_layout()
plt.savefig(r"C:\Users\Anton (Main)\Desktop\!extracted\sampleGalaxies\100008-modified1.jpg", bbox_inches='tight', pad_inches=0)
plt.show()


Grayimg = img_as_float(Transformed)
# gau_img = gaussian(image, sigma=2.25)
gau_img = gaussian(Grayimg, sigma=3)

#Display gaussian blurred image
array = np.reshape(gau_img, (72, 72))
plt.imshow(array, cmap=plt.cm.gray)
#plt.imshow(array)
plt.axis('off')
plt.show()