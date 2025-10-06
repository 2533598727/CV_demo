import cv2
import time

def read_phone_camera(stream_url):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("无法连接到摄像头，请检查网络、地址或身份验证！")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("画面获取失败，重试中...")
            time.sleep(1)
            cap = cv2.VideoCapture(stream_url)
            continue
        cv2.imshow('iPhone Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 用户名:密码@IP:端口/路径
    stream_url = "http://admin:12345@10.54.0.134:8081/video"  
    read_phone_camera(stream_url)