import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import PIL 
  
def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def parse_record(record):
    depth_major = record.reshape((3,32,32))
    image = np.transpose(depth_major,[0,1,2])
    return image

data_batch_1 = unpickle(r'C:\Users\kamuj\Downloads\cifar-10-batches-py\data_batch_1')

# image = data_batch_4[b'data'][56]
# labels = data_batch_4[b'labels']
# print(labels)
# plt.imshow(image)
# plt.show()

# img = Image.open(r"C:\Users\kamuj\Downloads\Figure.png")
# print("hel")
# image = np.array(img)
# print(img)

# image = image.reshape(3,32,32)
# image = image.transpose(1,2,0)
# plt.imshow(image)
# print(image.shape)
# image = image.save("Figure.png")

listIt = [29,30,35,4,5,32,6,13,18,9,17,21,3,10,20,27,40,51,0,19,23,7,11,12,8,8,8,1,2,14]
z = 0
for i in listIt:
    name="fig"+str(z)+".png"
    z+=1
    image= data_batch_1[b'data'][i]
    image = image.reshape(3,32,32)
    img=np.transpose(image,(1,2,0))
    Img=Image.fromarray(img,'RGB')
    Img.save(name)
