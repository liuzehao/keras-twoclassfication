# coding=utf-8
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
def get_all_files(bg_path):
    files = []

    for f in os.listdir(bg_path):
        if os.path.isfile(os.path.join(bg_path, f)):
            files.append(os.path.join(bg_path, f))
        else:
            files.extend(get_all_files(os.path.join(bg_path, f)))
    files.sort(key=lambda x: int(x[-10:-5]))#排序从小到大
    return files
def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
    # 2. 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
 
    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
 
    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations = 1)
 
    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations = 1)
    return erosion
val_path="./valdata"
label_val_path='./label_val.txt'
val_data=get_all_files(val_path)
val_images = np.empty((100000,40,100,1),dtype="float32")
val_labels= np.empty((100000,),dtype="int")
for i in range(len(val_data)):

    #1.载入图片 100*40
    # print(i)
    im=cv2.imread(val_data[i])
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# 2. 形态学变换的预处理，得到可以查找矩形的图片
    erosion = preprocess(gray)
    # cv2.imshow("img",erosion)
    # cv2.waitKey(0)
    arr = np.asarray(erosion, dtype="float32")
    arr.resize((40,100,1))
    val_images[i,:,:]=arr

#2.载入label
f = open(label_val_path, 'r')
lines = f.readlines()
for t in range(len(lines)):   
    val_labels[t]=int(lines[t])

model = load_model('./my_model.h5')
model.summary()
test_loss, test_acc = model.evaluate(val_images, val_labels)

print(test_acc)

