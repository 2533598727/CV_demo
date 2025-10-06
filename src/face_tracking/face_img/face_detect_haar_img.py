import cv2
import matplotlib.pyplot as plt
import numpy as np

# 方法1 显示图片
def show_img(img,title,pos):
    #BGR->RGB
    img_RGB = img[:,:,::-1]
    plt.subplot(2,2,pos)
    plt.title(title)
    plt.imshow(img_RGB)
    plt.axis('off')

# 方法2  绘制人脸矩形
def plot_rectangle(img,faces):
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),10)
    return img

#主函数
def main():
    #读取一张图片
    img = cv2.imread('src/face_tracking/face_img/input/2.PNG')
    
    #转化成灰度图片
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #通过cv2.CascadeClassifier加载分类器文件
    face_alt2 = cv2.CascadeClassifier('src/models/haarcascades/haarcascade_frontalface_alt2.xml')
    
    #对图片中的人脸进行检测
    faces = face_alt2.detectMultiScale(img_gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))

    #绘制检测到的人脸
    result = plot_rectangle(img.copy(),faces)

    #创建窗口并显示图片
    plt.figure(figsize=(9,9))
    plt.suptitle('Face Detection',fontsize=14,fontweight='bold')
    show_img(img,'Original Image',1)
    show_img(result,'Detection Image',2)
    plt.show()
if __name__ == '__main__':
    main()