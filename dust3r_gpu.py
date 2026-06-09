import os
import json
import base64
import argparse
import torch
import numpy as np
from glob import glob
from math import radians, cos, sqrt

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# ДОБАВИТЬ: импорт для создания mesh
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("[DUSt3R_GPU] Warning: open3d not installed, GLB generation disabled")


def haversine_distance_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1) * cos(radians((lat1 + lat2) / 2))
    return R * sqrt(dlat ** 2 + dlon ** 2)


def compute_scale_from_gps(gps_coords, cam_centers):
    """
    Returns (scale_weighted, scale_median) in m/unit.
    """
    N = len(gps_coords)
    if N < 2:
        return None, None

    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            d_gps  = haversine_distance_m(*gps_coords[i], *gps_coords[j])
            d_dust = np.linalg.norm(cam_centers[i] - cam_centers[j])
            if d_dust > 1e-6:
                pairs.append((d_gps, d_dust, d_gps / d_dust))

    if not pairs:
        return None, None

    all_r        = [r for _, _, r in pairs]
    scale_median = float(np.median(all_r))

    # Top-30% longest pairs (min 3), discard < 7m
    pairs.sort(key=lambda x: x[0], reverse=True)
    n_top    = max(3, len(pairs) * 30 // 100)
    top      = pairs[:n_top]
    GPS_MIN  = 7.0
    reliable = [(g, d, r) for g, d, r in top if g >= GPS_MIN]
    if not reliable:
        reliable = top
        print("[DUSt3R_GPU] Warning: all GPS pairs < 7m, scale accuracy is very low")

    weights        = np.array([g for g, _, _ in reliable])
    ratios         = np.array([r for _, _, r in reliable])
    scale_weighted = float(np.average(ratios, weights=weights))

    print(f"[DUSt3R_GPU] Scale weighted : {scale_weighted:.4f} m/unit "
          f"({len(reliable)}/{len(pairs)} pairs, "
          f"GPS {min(g for g,_,_ in reliable):.1f}..{max(g for g,_,_ in reliable):.1f}m)")
    print(f"[DUSt3R_GPU] Scale median   : {scale_median:.4f} m/unit "
          f"(all {len(all_r)} pairs, min={min(all_r):.3f} max={max(all_r):.3f})")
    print(f"[DUSt3R_GPU] GPS accuracy ~3-6m -> scale error ~60% -> volume error ~4x (scale^3)")

    return scale_weighted, scale_median


def focal_px_from_exif(exif_list, dust3r_img_size=512):
    for entry in exif_list:
        if not isinstance(entry, dict):
            continue
        e = entry.get("exif", entry)
        if not isinstance(e, dict):
            e = entry
        if not isinstance(e, dict):
            continue
        focal_real = e.get("focal_real")
        focal_35mm = e.get("focal_35mm")
        orig_w     = e.get("orig_width")
        orig_h     = e.get("orig_height")
        if not all(v is not None for v in [focal_real, focal_35mm, orig_w, orig_h]):
            continue
        if focal_real <= 0 or focal_35mm <= 0:
            continue
        aspect      = orig_w / orig_h
        crop_factor = focal_35mm / focal_real
        sensor_diag = 43.2666 / crop_factor
        sensor_w_mm = sensor_diag / sqrt(1 + aspect ** 2) * aspect
        fx_orig     = (focal_real / sensor_w_mm) * orig_w
        fx_dust3r   = fx_orig * (dust3r_img_size / max(orig_w, orig_h))
        print(f"[DUSt3R_GPU] EXIF focal: {focal_real}mm (35mm={focal_35mm}mm) "
              f"-> fx_dust3r={fx_dust3r:.1f}px")
        return fx_dust3r
    return None


def compute_volume_convex_hull(pts):
    from scipy.spatial import ConvexHull
    z_min    = pts[:, 2].min()
    mirrored = pts.copy()
    mirrored[:, 2] = 2 * z_min - mirrored[:, 2]
    hull = ConvexHull(np.vstack([pts, mirrored]))
    return hull.volume / 2.0


def save_ply(filepath, pts, colors):
    print(f"[DUSt3R_GPU] Saving {len(pts)} points to {filepath}...")
    with open(filepath, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])}\n")


# ДОБАВЛЕНА НОВАЯ ФУНКЦИЯ: конвертация point cloud в mesh (GLB) через Poisson
def pointcloud_to_mesh(pts, colors, output_glb_path, simplify=False, target_faces=20000):
    """
    Convert point cloud to mesh using Poisson Surface Reconstruction.
    Returns path to GLB file.
    """
    if not OPEN3D_AVAILABLE:
        print("[DUSt3R_GPU] open3d not installed, skipping mesh generation")
        return None
    
    print(f"[DUSt3R_GPU] Creating mesh from {len(pts)} points using Poisson...")
    
    # Создаем point cloud в open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    # 1. Удаляем выбросы (летящий мусор)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    print(f"[DUSt3R_GPU] After outlier removal: {len(pcd.points)} points")
    
    # 2. Вычисляем нормали (жизненно важно для Пуассона)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
    # 3. Реконструкция Пуассона (depth=9 дает хорошую детализацию, можно увеличить до 10)
    print("[DUSt3R_GPU] Running Poisson reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    # 4. Удаление лишней сетки (отрезаем "пузырь")
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # 5. Перенос цвета с точек на полигоны
    print("[DUSt3R_GPU] Transferring colors...")
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    mesh_colors = []
    for v in np.asarray(mesh.vertices):
        _, idx, _ = kdtree.search_knn_vector_3d(v, 1)
        mesh_colors.append(np.asarray(pcd.colors)[idx[0]])
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack(mesh_colors))
    
    # 6. --- ФИКС ПЕРЕВЕРНУТОСТИ ---
    # Поворачиваем меш на 180 градусов вокруг оси X.
    R = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(R, center=mesh.get_center())

    # 7. --- ФИКС "БЕЛЕСОСТИ" И МАТОВОСТИ (Гамма-коррекция) ---
    # Делаем цвета чуть темнее и насыщеннее (возводим в степень 1.5)
    mesh_colors_np = np.asarray(mesh.vertex_colors)
    mesh_colors_np = np.power(mesh_colors_np, 1.5) 
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors_np)
    
    # 8. Сохраняем как GLB
    o3d.io.write_triangle_mesh(output_glb_path, mesh, write_ascii=False, compressed=True)
    print(f"[DUSt3R_GPU] Mesh saved to {output_glb_path} ({len(mesh.triangles)} faces)")
    
    return output_glb_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--exif_json", default=None)
    # ДОБАВЛЕН НОВЫЙ ПАРАМЕТР: генерировать ли GLB
    parser.add_argument("--no_glb",    action="store_true", help="Skip GLB mesh generation")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[DUSt3R_GPU] Device: {device.upper()}")

    # 1. Model
    ckpt_paths = glob("checkpoints/*.pth")
    if not ckpt_paths:
        raise FileNotFoundError("No checkpoint in checkpoints/")
    model = AsymmetricCroCo3DStereo.from_pretrained(ckpt_paths[0]).to(device)
    print(f"[DUSt3R_GPU] Model: {ckpt_paths[0]}")

    # 2. Images
    img_files = sorted(
        glob(os.path.join(args.image_dir, "*.jpg")) +
        glob(os.path.join(args.image_dir, "*.png"))
    )
    if len(img_files) < 2:
        raise ValueError(f"Need at least 2 images, found {len(img_files)}")
    print(f"[DUSt3R_GPU] Images: {len(img_files)}")
    imgs = load_images(img_files, size=512)

    # 3. Parse EXIF
    gps_coords  = None
    known_focal = None
    if args.exif_json and os.path.exists(args.exif_json):
        with open(args.exif_json) as f:
            exif_list = json.load(f)
        known_focal = focal_px_from_exif(exif_list, dust3r_img_size=512)
        sorted_exif = sorted(exif_list, key=lambda e: e.get("photo_index", 0))
        gps_list = []
        for entry in sorted_exif:
            if not isinstance(entry, dict):
                continue
            inner = entry.get("exif", entry)
            if not isinstance(inner, dict):
                inner = entry
            if not isinstance(inner, dict):
                continue
            lat = inner.get("lat")
            lon = inner.get("lon")
            if lat is not None and lon is not None:
                gps_list.append((lat, lon))
        if len(gps_list) == len(img_files):
            gps_coords = gps_list
            print(f"[DUSt3R_GPU] GPS loaded: {len(gps_coords)} points")
        else:
            print(f"[DUSt3R_GPU] GPS mismatch: {len(gps_list)} vs {len(img_files)} images")

    # 4. Inference
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    print("[DUSt3R_GPU] Running inference...")
    output = inference(pairs, model, device, batch_size=2)

    # 5. Global Alignment
    print("[DUSt3R_GPU] Global Alignment...")
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=0.01)

    try:
        opt_focals = scene.get_focals().detach().cpu().numpy().flatten()
        print(f"[DUSt3R_GPU] Optimized focals: mean={opt_focals.mean():.1f}px "
              f"spread={opt_focals.min():.1f}..{opt_focals.max():.1f}px")
        if known_focal:
            diff_pct = abs(opt_focals.mean() - known_focal) / known_focal * 100
            print(f"[DUSt3R_GPU] EXIF focal={known_focal:.1f}px  diff={diff_pct:.1f}%")
    except Exception:
        pass

    # 6. Point cloud
    pts3d    = scene.get_pts3d()
    masks    = scene.get_masks()
    rgb_imgs = scene.imgs

    pts3d_np = [p.detach().cpu().numpy() for p in pts3d]
    masks_np = [m.detach().cpu().numpy() for m in masks]

    all_pts, all_colors = [], []
    total_before = sum(m.size for m in masks_np)
    total_after  = sum(m.sum() for m in masks_np)
    print(f"[DUSt3R_GPU] Masks: {total_after}/{total_before} points passed confidence filter")
    for i in range(len(pts3d_np)):
        mask = masks_np[i]
        all_pts.append(pts3d_np[i][mask])
        c = (rgb_imgs[i][mask] * 255.0).clip(0, 255).astype(np.uint8)
        all_colors.append(c)

    final_pts    = np.concatenate(all_pts,    axis=0)
    final_colors = np.concatenate(all_colors, axis=0)

    # 7. GPS scale (both weighted and median)
    scale_weighted = None
    scale_median   = None
    if gps_coords is not None:
        poses       = scene.get_im_poses().detach().cpu().numpy()
        cam_centers = poses[:, :3, 3]
        scale_weighted, scale_median = compute_scale_from_gps(gps_coords, cam_centers)

    # 8. Volume — compute with both scales
    volume_units   = None
    volume_m3      = None   # using weighted scale
    volume_m3_med  = None   # using median scale

    try:
        volume_units = compute_volume_convex_hull(final_pts)
        print(f"[DUSt3R_GPU] Volume (DUSt3R units): {volume_units:.6f}")

        if scale_weighted is not None:
            volume_m3     = volume_units * (scale_weighted ** 3)
            volume_m3_med = volume_units * (scale_median   ** 3)
            print(f"[DUSt3R_GPU] Volume weighted scale : {volume_m3:.3f} m3")
            print(f"[DUSt3R_GPU] Volume median scale   : {volume_m3_med:.3f} m3")
            print(f"[DUSt3R_GPU] WARNING: GPS error ~60% -> volume error up to 4x")
        else:
            print("[DUSt3R_GPU] No GPS -> volume in relative units only")

    except ImportError:
        print("[DUSt3R_GPU] scipy missing: pip install scipy --break-system-packages")
    except Exception as e:
        print(f"[DUSt3R_GPU] Volume error: {e}")

    # 9. Save PLY
    save_ply(args.output, final_pts, final_colors)
    
    # 10. Read PLY as base64 to pass back via JSON
    ply_base64 = None
    try:
        with open(args.output, "rb") as f:
            ply_base64 = base64.b64encode(f.read()).decode("utf-8")
        print(f"[DUSt3R_GPU] PLY encoded to base64 ({len(ply_base64)//1024} KB)")
    except Exception as e:
        print(f"[DUSt3R_GPU] PLY base64 error: {e}")
    
    # 11. ДОБАВЛЕНО: Generate GLB mesh
    glb_base64 = None
    glb_file_path = None
    
    if not args.no_glb and OPEN3D_AVAILABLE and len(final_pts) > 1000:
        try:
            glb_file_path = args.output.replace(".ply", ".glb")
            pointcloud_to_mesh(final_pts, final_colors, glb_file_path, simplify=True)
            
            # Read and encode GLB as base64
            with open(glb_file_path, "rb") as f:
                glb_base64 = base64.b64encode(f.read()).decode("utf-8")
            print(f"[DUSt3R_GPU] GLB encoded to base64 ({len(glb_base64)//1024} KB)")
        except Exception as e:
            print(f"[DUSt3R_GPU] GLB generation error: {e}")
            glb_base64 = None
    elif not args.no_glb and not OPEN3D_AVAILABLE:
        print("[DUSt3R_GPU] Skipping GLB: open3d not available")
    elif not args.no_glb and len(final_pts) <= 1000:
        print(f"[DUSt3R_GPU] Skipping GLB: too few points ({len(final_pts)}) for mesh")

    # 12. Result JSON (ДОБАВЛЕНО поле glb_base64)
    result = {
        "status":              "success",
        "point_count":         len(final_pts),
        "scale_weighted":      round(scale_weighted, 4) if scale_weighted else None,
        "scale_median":        round(scale_median,   4) if scale_median   else None,
        "volume_dust3r_units": round(volume_units,   6) if volume_units is not None else None,
        "volume_m3_weighted":  round(volume_m3,      3) if volume_m3      is not None else None,
        "volume_m3_median":    round(volume_m3_med,  3) if volume_m3_med  is not None else None,
        "ply_file":            args.output,
        "ply_base64":          ply_base64,
        "glb_file":            glb_file_path,      # ДОБАВЛЕНО
        "glb_base64":          glb_base64          # ДОБАВЛЕНО
    }
    result_path = args.output.replace(".ply", "_result.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[DUSt3R_GPU] Result: {result_path}")
    # Print result without base64 blobs for clean logs
    log_result = {k: v for k, v in result.items() if k not in ["ply_base64", "glb_base64"]}
    print(json.dumps(log_result))
    print("[DUSt3R_GPU] Done!")


if __name__ == '__main__':
    main()
