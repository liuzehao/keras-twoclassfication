#conding=utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
train_path="./traindata"
label_train_path='label_train.txt'
val_path="./valdata"
label_val_path='./label_val.txt'
if __name__=='__main__':
    train_data=get_all_files(train_path)
    train_images = np.empty((10000,40,100,1),dtype="float32")
    train_labels= np.empty((10000,),dtype="int")
    for i in range(len(train_data)):

        #1.载入图片 100*40
        # print(i)
        im=cv2.imread(train_data[i])
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
 
    # 2. 形态学变换的预处理，得到可以查找矩形的图片
        erosion = preprocess(gray)
        # cv2.imshow("img",erosion)
        # cv2.waitKey(0)
        arr = np.asarray(erosion, dtype="float32")
        arr.resize((40,100,1))
        train_images[i,:,:]=arr
    
    #2.载入label
    f = open(label_train_path, 'r')
    lines = f.readlines()
    for t in range(len(lines)):   
        train_labels[t]=int(lines[t])
       # print(train_labels[t])
    #print(train_labels)
    #print(lines)
    # 3.构造模型

    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(40,100,1)),
    #     keras.layers.Dense(128, activation=tf.nn.relu),
    #     keras.layers.Dense(1, activation=tf.nn.softmax)
    # ])
    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    #model.add(layers.Conv2D(64, (3, 3), padding='same',input_shape=(40, 100, 1) ,activation='relu'))
    model.add(layers.Flatten(input_shape=(40,100,1)))
    model.add(layers.Dense(128, activation='relu'))
    # Add another:
    model.add(layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    #4.训练
    model.fit(train_images, train_labels, epochs=10)
    #5.测试
    model.save('my_model.h5')
    del model