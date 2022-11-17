from PIL import Image
from torchvision import transforms
# from skimage import data, img_as_float

#Images: \J010\J010002.98-045610.5.png , J011\J011001.20+002441.9.png , J011\J011002.57+050018.4.png , J011\J011002.82+002515.1.png , J092928.60+231256.2.png
#Blank image J095628.63+244102.5.png
img = Image.open(r"C:\Users\Anton (Main)\Desktop\!extracted\sampleGalaxies\100008-modified.jpg")

transform=transforms.Compose([transforms.ToTensor(), transforms.ToPILImage()])

IMG = transform(img)
clrs = IMG.getcolors()

#This prints the number of different colours in the image, if equal to 1 the image is all one colour (it may be blank)
print(clrs)