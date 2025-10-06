import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

#方法1 显示图片
def show_img(img,title,pos):
    # 显示图片
    img_RGB = img[:,:,::-1]
    plt.subplot(2,2,pos)
    plt.title(title)
    plt.imshow(img_RGB)
    plt.axis('off')


#方法2 绘制矩形框
def plot_rectangle(img,faces):
    for face in faces:
        cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 5)
    return img
def main():
    # 读取图片
    img = cv2.imread('src/face_tracking/face_img/input/2.PNG')
    # 灰度转换
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #调用dlib检测器
    detection = dlib.get_frontal_face_detector()
    faces = detection(img_gray,1) # 1:放大倍数
    #绘制出人脸矩形框
    result = plot_rectangle(img.copy(), faces)
    #显示图片
    plt.figure(figsize=(9,9))
    plt.suptitle('Face Detection',fontsize=14,fontweight='bold') 
    show_img(img,'Original Image',1) # 原图
    show_img(result,'Detection Image',2) # 检测后的图
    plt.show()

if __name__ == '__main__':
    main()
