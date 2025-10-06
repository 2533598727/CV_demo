import cv2
import dlib

#方法2 绘制矩形框
def plot_rectangle(img,faces):
    for face in faces:
        cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 5)
    return img


def main(stream_url):
    # 读取摄像头
    cap = cv2.VideoCapture(stream_url)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
    fps= int(cap.get(cv2.CAP_PROP_FPS))
    print("高度:{}".format(height))
    print("宽度:{}".format(width))
    print("帧率:{}".format(fps))

    #调用dlib检测器
    detection = dlib.get_frontal_face_detector()
    

    while True:
        ret, frame = cap.read()
        #检验摄像头
        if not ret or frame is None:
            break
        if ret == True:
            #灰度转换
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #调用检测器
        faces = detection(gray_frame.copy(),1)
        frame = plot_rectangle(frame.copy(),faces)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

  

if __name__ == '__main__':
    stream_url = "http://admin:12345@10.54.0.134:8081/video"  
    main(stream_url)