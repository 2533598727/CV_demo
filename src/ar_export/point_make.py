import numpy as np
from PIL import Image
import cv2 
import open3d as o3d
import os
import zipfile  
from pxr import Usd, UsdGeom, Sdf, Vt  # 生成usdz
import trimesh                         # 3D预览


def save_pcd_as_usdz(pcd, usdz_path, point_size=0.003):
    print(f"\n正在将点云导出为 USDZ: {usdz_path}")
    file_base_name = os.path.splitext(os.path.basename(usdz_path))[0]
    usdc_file_name = f"{file_base_name}.usdc" 
    
    points = np.asarray(pcd.points)
    num_points = len(points)
    has_colors = pcd.has_colors()
    if has_colors:
        colors = np.asarray(pcd.colors)

    print(f"   - 1. 正在创建临时的二进制 .usdc 场景文件: {usdc_file_name}")
    stage = Usd.Stage.CreateNew(usdc_file_name)
    UsdGeom.SetStageUpAxis(stage, 'Y')
    
    points_prim = UsdGeom.Points.Define(stage, '/World/PointCloud')
    points_prim.GetPointsAttr().Set(Vt.Vec3fArray(points.tolist()))

    widths_attr = points_prim.CreateWidthsAttr()
    widths_attr.Set(Vt.FloatArray([point_size] * num_points))
    print(f"   - 已将 {num_points} 个点的大小统一设置为 {point_size}")

    if has_colors:
        color_primvar = points_prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        color_primvar.Set(Vt.Vec3fArray(colors.tolist()))

    stage.Save()

    print(f"   - 2.正在将 {usdc_file_name} 打包为 USDZ...")
    try:
        with zipfile.ZipFile(usdz_path, 'w', zipfile.ZIP_STORED) as zf:
            zf.write(usdc_file_name, arcname=usdc_file_name)
        print(f"USDZ 文件已成功创建！")
        return True
    except Exception as e:
        print(f"   - 手动打包过程中发生错误: {e}")
        return False
    finally:
        if os.path.exists(usdc_file_name):
            print(f"   - 3. 清理临时文件: {usdc_file_name}")
            os.remove(usdc_file_name)




if __name__ == '__main__':
    color_image_path = "src/ar_export/input/style_transfer_result.png"
    depth_map_path = "src/ar_export/output/depth_map.npy" 

    ply_output_path = "src/ar_export/output/style_transfer_point_cloud_img.ply"
    usdz_output_path = "src/ar_export/output/style_transfer_point_cloud_img.usdz"
    

    print("1. 正在加载图像和深度图...")
    color_raw = Image.open(color_image_path)
    depth_raw = np.load(depth_map_path)
    
    if color_raw.size != (depth_raw.shape[1], depth_raw.shape[0]):
        depth_resized = cv2.resize(depth_raw, color_raw.size, interpolation=cv2.INTER_LINEAR)
    else:
        depth_resized = depth_raw
    
    #深度反转调整 z小-近 z大-远
    normalized_depth = (depth_resized - np.min(depth_resized)) / (np.max(depth_resized) - np.min(depth_resized))
    inverted_depth = 1.0 - normalized_depth
    
    print("2. 正在根据深度图构建初始点云...")
    width, height = color_raw.size
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    depth_strength = -100.0
    z_coords = inverted_depth * depth_strength
    points = np.vstack((u.flatten(), v.flatten(), z_coords.flatten())).T
    colors = np.array(color_raw).reshape(-1, 3) / 255.0 # 归一化

    print("3. 正在应用艺术化过滤算法...")
    overall_density = 0.8  #整体密度
    brightness_weight = 0.8  #亮区权重-越亮保留概率越高
    detail_weight = 0.9      #细节权重-边缘/纹理区域更容易保留

    gray_img = cv2.cvtColor(np.array(color_raw), cv2.COLOR_RGB2GRAY)
    brightness_prob = gray_img / 255.0

    # 拉普拉斯算子-检测边缘
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    detail_prob = np.abs(laplacian) / np.max(np.abs(laplacian))

    #最终概率：亮度×亮度权重 + 细节×细节权重
    final_probabilities = (brightness_prob * brightness_weight) + (detail_prob * detail_weight)
    final_probabilities *= overall_density

    #随机生成筛选点
    mask = (np.random.rand(height, width) < final_probabilities).flatten()
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    print(f"   - 筛选前总点数: {len(points)}")
    print(f"   - (密度参数: {overall_density}) 筛选后保留点数: {len(filtered_points)}")

    print("4. 正在创建最终的 Open3D 对象...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    pcd.translate(-pcd.get_center(), relative=True)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    print("- 正在归一化点云尺寸...")
    bbox = pcd.get_axis_aligned_bounding_box()
    max_extent = np.max(bbox.get_extent())
    if max_extent > 1e-6: 
        target_max_dimension = 1.0
        scale_factor = target_max_dimension / max_extent
        pcd.scale(scale_factor, center=pcd.get_center())
        print(f"   - 点云已从最大尺寸 {max_extent:.2f} 缩放到 {target_max_dimension:.2f}")

    print("\n5.正在保存文件")
    o3d.io.write_point_cloud(ply_output_path, pcd) 
    print(f"PLY文件已保存: {ply_output_path}")


    success = save_pcd_as_usdz(pcd, usdz_output_path, point_size=0.003)
    print("\n6. 正在显示 Open3D 预览窗口，按 'q' 关闭。")
    o3d.visualization.draw_geometries([pcd])