import cv2

# 方法2  绘制人脸矩形
def plot_rectangle(img,faces):
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),10)
    return img

#主函数
def main(stream_url):
    #读取摄像头
    cap = cv2.VideoCapture(stream_url)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
    fps= int(cap.get(cv2.CAP_PROP_FPS))
    print("高度:{}".format(height))
    print("宽度:{}".format(width))
    print("帧率:{}".format(fps))


    #通过cv2.CascadeClassifier加载分类器文件
    face_alt2 = cv2.CascadeClassifier('src/models/haarcascades/haarcascade_frontalface_alt2.xml')
    

    while True:
        ret, frame = cap.read()
        #检验摄像头
        if not ret or frame is None:
            break
        if ret == True:
            #灰度转换
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #对图片中的人脸进行检测
        faces = face_alt2.detectMultiScale(gray_frame.copy(),scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
        
        frame = plot_rectangle(frame.copy(),faces)
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    stream_url = "http://admin:12345@10.54.0.134:8081/video"  
    main(stream_url)