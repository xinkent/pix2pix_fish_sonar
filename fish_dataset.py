import numpy as np
from PIL import Image

from io import BytesIO


def load_dataset(dataDir='./dataset/train_data/', data_range=range(0,300)):
        print("load dataset start")
        print("     from: %s"%dataDir)
        imgDataset = []
        labelDataset = []

        imgStart = 1515
        labelStart = 1
        for i in data_range:
            imgNum = imgStart + int(i*(29/10))
            labelNum = i + 1
            img = Image.open(dataDir + "GP030023_%06d.png"%imgNum)
            label = Image.open(dataDir + "2017-03-02_105804_%d_width_773_height_1190.png"%labelNum)
            label = label.convert("L")
            img = img.resize((512,256), Image.BILINEAR)
            img= img.transpose(Image.ROTATE_90)
            label = label.resize((256, 512),Image.BILINEAR)

            img = np.asarray(img)/128.0-1.0

            label = np.asarray(label)/128.0-1.0
            label = label[:,:,np.newaxis]

            imgDataset.append(img)
            labelDataset.append(label)

        print("load dataset done")
        return np.array(imgDataset),np.array(labelDataset)
