# 你可以在你的主脚本之后添加这段代码，或者创建一个新脚本
import torch
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False

def get_depth_map(img_path, model_type="DPT_Large"):
    # 加载MiDaS模型
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # 加载MiDaS的图像转换器
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if model_type == "DPT_Large" or model_type == "DPT_Hybrid" else midas_transforms.small_transform

    # 加载并处理图像
    if img_path.startswith(('http', 'https')):
        response = requests.get(img_path)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(img_path).convert("RGB")
    
    img_np = np.array(img)
    input_batch = transform(img_np).to(device)

    with torch.no_grad(): #关闭梯度计算
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_np.shape[:2],
            mode="bicubic", #双三次线性差值
            align_corners=False,#取消强制对其边缘像素
        ).squeeze() # 通道数 高 宽

    
    depth_map = prediction.cpu().numpy()
 
    # 1. 先将深度图归一化到 0-1 的范围
    depth_normalized_float = cv2.normalize(depth_map, 
        None,
        1.0,0.0, 
        cv2.NORM_MINMAX,
        cv2.CV_32F) #32位浮点数

    # 2. 转换为 8-bit 整数图，以便使用equalizeHist
    depth_uint8 = (depth_normalized_float * 255).astype(np.uint8) 
    # 3. 应用直方图均衡化
    depth_enhanced_uint8 = cv2.equalizeHist(depth_uint8)
    # 4. 将增强后的结果转换回 0-1 的浮点数范围，作为最终的深度图
    final_depth_map = depth_enhanced_uint8.astype(np.float32) / 255.0
    

    output_colored = cv2.applyColorMap(depth_enhanced_uint8, cv2.COLORMAP_INFERNO)
    return final_depth_map, output_colored



content_img_path = "src/ar_export/input/style_transfer_result.png"
#content_img_path = "src/style_transfer/input/IMG_C4.JPG"
depth_map, depth_viz = get_depth_map(content_img_path)

# 可视化深度图
plt.imshow(depth_viz)
plt.title("预测的深度图")
plt.show()

# 保存深度图数据
np.save("src/ar_export/output/depth_map.npy", depth_map)
cv2.imwrite("src/ar_export/output/depth_visualization.png", depth_viz)

# 可视化深度图
plt.imshow(cv2.cvtColor(depth_viz, cv2.COLOR_BGR2RGB)) # plt显示需要RGB
plt.title("预测的深度图 (已保存为PNG)")
plt.show()
