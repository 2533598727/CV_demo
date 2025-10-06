import torch
import torchvision
from torch import nn #神经网络模块 
from torchvision import transforms #图像预处理
from PIL import Image as PILImage 
import matplotlib.pyplot as plt
import os
import requests #网络请求读取url
from io import BytesIO #内存中处理图像字节流
import cv2
import numpy as np
import time

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False


def load_image(img_path_or_url): #加载图像
    if img_path_or_url.startswith(('http://', 'https://')):#从url加载图像
        try:
            response = requests.get(img_path_or_url, timeout=10)
            response.raise_for_status() #检查响应状态码
            img = PILImage.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"从URL加载图像失败: {e}")
            raise
    else:
        if not os.path.exists(img_path_or_url):
            raise FileNotFoundError(f"图像文件不存在: {img_path_or_url}")
        img = PILImage.open(img_path_or_url).convert('RGB')
    
    return img

def preprocess(img, image_shape, augment=False):#预处理
    if isinstance(img, np.ndarray):
        if len(img.shape) == 3 and img.shape[2] == 3:  #三通道彩色图像
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = PILImage.fromarray(img)
    
    transforms_list = [
        transforms.Resize(image_shape),
        transforms.ToTensor(),
    ]
    
    if augment:#数据增强
        transforms_list.insert(1, transforms.RandomResizedCrop(image_shape, scale=(0.9, 1.0)))
        transforms_list.insert(2, transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
    
    transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    transform = transforms.Compose(transforms_list) #合成管道
    return transform(img).unsqueeze(0) # 批次 通道数 高度 宽度

def postprocess(img_tensor): #还原图像数据
    img = img_tensor.clone().detach().cpu().squeeze() #分离张量 移动到cpu 移除批次维度 
    img = img.permute(1, 2, 0) #高度 宽度 通道数
    
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img = img * std + mean #反归一化
    
    img = torch.clamp(img, 0, 1)
    img_np = img.numpy()
    
    return img_np

def extract_features(X_input, content_layers, style_layers, net): #特征提取
 
    contents = [] 
    styles = []
    x = X_input #输入张量

    for i in range(len(net)): 
        x = net[i](x) 
        if i in style_layers:
            styles.append(x)
        if i in content_layers:
            contents.append(x)
    return contents, styles

def get_contents(content_img, image_shape, device, content_layers, style_layers, net):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers, net)
    return content_X, contents_Y

def get_styles(style_img, image_shape, device, content_layers, style_layers, net):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers, net)
    return style_X, styles_Y

def content_loss(Y_hat, Y, weight=1.0):#内容损失
    return torch.square(Y_hat - Y.detach()).mean() * weight #MSE

def style_loss(Y_hat, Y_gram, weight=1.0):#风格损失
    Y_hat_gram = gram(Y_hat) #Gram捕捉风格信息
    return torch.square(Y_hat_gram - Y_gram.detach()).mean() * weight

def gram(X):
    # 批次 通道数 高度 宽度
    num_channels, n = X.shape[1], X.numel() // X.shape[1] # n-每个通道的元素数量
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n + 1e-8) #计算向量内积并归一化

def tv_loss(Y_hat, weight=10.0):#总变差损失
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean()) * weight #xy方向 相邻像素变化程度

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram): #计算总损失
    content_weight = 1
    style_weight = 1e4 #风格权重
    tv_weight = 1
    
    content_weights = [content_weight] * len(contents_Y_hat)
    # 风格损失：随着层数的增加，权重逐渐减小
    style_weights = [style_weight * (1.0 / (i + 1)) for i in range(len(styles_Y_hat))]

    contents_l = [content_loss(Y_hat, Y, w) for Y_hat, Y, w in zip(contents_Y_hat, contents_Y, content_weights)]
    styles_l = [style_loss(Y_hat, Y, w) for Y_hat, Y, w in zip(styles_Y_hat, styles_Y_gram, style_weights)]
    tv_l = tv_loss(X, tv_weight)
    
    l = sum(contents_l + styles_l + [tv_l])
    return contents_l, styles_l, tv_l, l

def get_inits(X, device, lr, styles_Y): #初始化
    gen_img = X.clone().requires_grad_(True) #标记需要的计算梯度

    #初始化优化器 lr-学习率 betas eps 超参数
    trainer = torch.optim.Adam([gen_img], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    #预计算
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img, styles_Y_gram, trainer

def resize_tensor(tensor, new_shape): #调整张量大小 双线性插值法
    return nn.functional.interpolate(tensor, size=new_shape, mode='bilinear', align_corners=False)

def train(X, contents_Y, styles_Y, style_X, device, lr, num_epochs, lr_decay_epoch, content_layers, style_layers, net):
    original_content_X = X.clone()
    original_style_X = style_X.clone()
    
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    
    #学习率衰减策略 监控总损失 当连续20个epoch没有改善时 降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer, mode='min', factor=0.5, patience=20
    )
    
    plt.figure(figsize=(15, 4))
    best_loss = float('inf')
    best_result = None
    
    start_time = time.time()
    
    #多尺度训练设置 逐步提高分辨率
    use_multi_scale = True
    scale_schedule = {
        0: (300, 400),
        200: (400, 500),
        400: (500, 600)
    }
    
    #迭代
    for epoch in range(num_epochs):
        if use_multi_scale and epoch in scale_schedule: #在指定分层 调整尺寸 
            new_shape = scale_schedule[epoch]
            if X.shape[2:4] != new_shape:
                print(f"\n调整训练尺度: {X.shape[2:4]} -> {new_shape}")
            
            resized_X = resize_tensor(X, new_shape).detach().requires_grad_(True)
            X = resized_X
            #重新初始化优化器
            trainer = torch.optim.Adam([X], lr=trainer.param_groups[0]['lr'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                trainer, mode='min', factor=0.5, patience=20
            )
            
            
            resized_content_X = resize_tensor(original_content_X, new_shape)
            contents_Y, _ = extract_features(resized_content_X, content_layers, style_layers, net)
            
            resized_style_X = resize_tensor(original_style_X, new_shape)
            _, new_styles_Y = extract_features(resized_style_X, content_layers, style_layers, net)
            styles_Y_gram = [gram(Y) for Y in new_styles_Y]
        
        #梯度清零
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers, net)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward() #反向传播
        
        torch.nn.utils.clip_grad_norm_([X], max_norm=1.0)#梯度裁剪
        
        trainer.step()
        scheduler.step(l)
        
        if l < best_loss:#保存最好的结果
            best_loss = l
            best_result = X.clone()
        
        if (epoch + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f'epoch {epoch + 1}/{num_epochs}, 内容损失: {sum(contents_l):.3f}, ' \
                  f'风格损失: {sum(styles_l):.3f}, 总变差损失: {tv_l:.3f}, 总损失: {l:.3f}, 耗时: {elapsed_time:.2f}s')
            
            if (epoch + 1) % 50 == 0:
                plot_index = (epoch + 1) // 50
                if plot_index <= 12: # Ensure we don't exceed subplot count
                    plt.subplot(1, 12, plot_index)
                    plt.title(f'E{epoch + 1}')
                    plt.imshow(postprocess(X))
                    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return best_result if best_result is not None else X

def main():
    #优先使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    image_shape = (300, 400)

    content_img_path = "src/style_transfer/input/IMG_C4.JPG"
    style_img_path = "src/style_transfer/input/IMG_S.JPG"
        
    content_img = load_image(content_img_path)
    style_img = load_image(style_img_path)
        
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"内容图像 (尺寸: {content_img.size[0]}x{content_img.size[1]})")
    plt.imshow(content_img)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title(f"风格图像 (尺寸: {style_img.size[0]}x{style_img.size[1]})")
    plt.imshow(style_img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
        
        #加载VGG19模型  加载预训练权重 不需要分类器 确保在device上运行
    vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features.to(device)
        
    #冻结VGG19模型参数 不参与训练    
    for param in vgg.parameters():
        param.requires_grad_(False)

    # 选择需要提取的层    
    style_layers = [0, 5, 10, 19, 28]
    content_layers = [25]
        
    #提取图像特征
    content_X, contents_Y = get_contents(content_img, image_shape, device, content_layers, style_layers, vgg)
    style_X, styles_Y = get_styles(style_img, image_shape, device, content_layers, style_layers, vgg)
        
    lr = 0.3
    num_epochs = 1200
    lr_decay_epoch = 50
        
    print("开始风格迁移训练...")
    print(f"训练配置: 学习率={lr}, 迭代次数={num_epochs}, 设备={device}")
    output = train(content_X, contents_Y, styles_Y, style_X, device, lr, num_epochs, lr_decay_epoch, content_layers, style_layers, vgg)
        
    result_np = postprocess(output)
    result_pil = PILImage.fromarray((result_np * 255).astype(np.uint8))
    result_path = "src/style_transfer/output/style_transfer_result.png"
    result_pil.save(result_path)
    print(f"结果已保存至: {result_path}")
        
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("内容图像")
    plt.imshow(content_img)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title("风格图像")
    plt.imshow(style_img)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title("优化后的风格迁移结果")
    plt.imshow(result_np)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print("风格迁移完成！")
        

if __name__ == '__main__':
    main()
