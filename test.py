import os
from PIL import Image
from pylab import *

path = '/home/lucas/Lab/SRGAN/SRGAN/data/person/val/'
files = os.listdir(path)
count=0
for file in files:
    count+=1
    print(file)
    try:
        image=array(Image.open(path+file))
        print(image.shape)
        if image.shape[0]<240 or image.shape[1]<240:
            print('%s is too small.' %file)
            os.remove(path + file)
        elif image.shape[0]==300 and image.shape[1]==400:
            os.remove(path + file)
        elif image.shape[0]>900 and image.shape[1]>900:
            os.remove(path + file)
    except Exception as e:
        print(e)
        print('cannot open file %s.' %file)
        os.remove(path+file)
print(count)