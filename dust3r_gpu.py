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

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("[DUSt3R_GPU] Warning: open3d not installed, GLB generation and isolation disabled")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[DUSt3R_GPU] Warning: opencv not installed, cube scale detection disabled")


# ---------------------------------------------------------------------------
# GPS scale
# ---------------------------------------------------------------------------

def haversine_distance_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1) * cos(radians((lat1 + lat2) / 2))
    return R * sqrt(dlat ** 2 + dlon ** 2)

def compute_scale_from_gps(gps_coords, cam_centers):
    """Returns (scale_weighted, scale_median) in m/unit."""
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

    return scale_weighted, scale_median


# ---------------------------------------------------------------------------
# Cube scale
# ---------------------------------------------------------------------------

def compute_scale_from_cube(img_files, pts3d_np, masks_np,
                             square_size_m=0.02,
                             pattern=(3, 3),
                             dust3r_size=512):
    if not CV2_AVAILABLE:
        print("[DUSt3R_GPU] OpenCV not available, skipping cube scale")
        return None

    all_scales = []

    for img_idx, img_path in enumerate(img_files):
        img = cv2.imread(img_path)
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, pattern, None)
        if not ret:
            small = cv2.resize(gray, (orig_w // 2, orig_h // 2))
            ret, corners_small = cv2.findChessboardCorners(small, pattern, None)
            if ret:
                corners = corners_small * 2.0
            else:
                continue

        if ret:
            corners = cv2.cornerSubPix(
                gray, corners.astype(np.float32), (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

        pts_map  = pts3d_np[img_idx]
        mask_map = masks_np[img_idx]
        dust3r_h, dust3r_w = pts_map.shape[:2]

        scale_x = dust3r_w / orig_w
        scale_y = dust3r_h / orig_h

        corners_flat = corners.reshape(-1, 2)
        pts3d_corners = []
        valid_flags   = []

        for (px, py) in corners_flat:
            ix = int(np.clip(px * scale_x, 0, dust3r_w - 1))
            iy = int(np.clip(py * scale_y, 0, dust3r_h - 1))
            patch_r = 2
            best_pt = None
            for dy in range(-patch_r, patch_r + 1):
                for dx in range(-patch_r, patch_r + 1):
                    ny = int(np.clip(iy + dy, 0, dust3r_h - 1))
                    nx = int(np.clip(ix + dx, 0, dust3r_w - 1))
                    if mask_map[ny, nx]:
                        best_pt = pts_map[ny, nx]
                        break
                if best_pt is not None:
                    break

            if best_pt is not None:
                pts3d_corners.append(best_pt)
                valid_flags.append(True)
            else:
                pts3d_corners.append(pts_map[iy, ix])
                valid_flags.append(True)
        
        cols = pattern[0]
        rows = pattern[1]
        neighbor_pairs = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if c + 1 < cols:
                    neighbor_pairs.append((idx, idx + 1))
                if r + 1 < rows:
                    neighbor_pairs.append((idx, idx + cols))

        img_scales = []
        for (i, j) in neighbor_pairs:
            if valid_flags[i] and valid_flags[j]:
                d_dust = np.linalg.norm(pts3d_corners[i] - pts3d_corners[j])
                if d_dust > 1e-6:
                    img_scales.append(square_size_m / d_dust)

        # -- ФИЛЬТРАЦИЯ ВЫБРОСОВ В ПАРАХ КУБА --
        if img_scales:
            img_scales_arr = np.array(img_scales)
            img_median = float(np.median(img_scales_arr))
            
            # Отсекаем пары, которые сильно выбиваются из медианы (шум детекции)
            valid_scales = img_scales_arr[(img_scales_arr >= 0.5 * img_median) & (img_scales_arr <= 2.0 * img_median)]
            
            if len(valid_scales) > 0:
                img_scale = float(np.median(valid_scales))
                all_scales.append(img_scale)
                print(f"[DUSt3R_GPU] Cube found on photo {img_idx}: scale={img_scale:.4f} m/unit "
                      f"(filtered {len(valid_scales)}/{len(img_scales)} valid pairs)")

    if not all_scales:
        print("[DUSt3R_GPU] Cube not found on any photo")
        return None

    scale_cube = float(np.median(all_scales))
    print(f"[DUSt3R_GPU] Cube scale FINAL: {scale_cube:.4f} m/unit")
    return scale_cube


# ---------------------------------------------------------------------------
# EXIF focal
# ---------------------------------------------------------------------------

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
        print(f"[DUSt3R_GPU] EXIF focal: {focal_real}mm (35mm={focal_35mm}mm) -> fx={fx_dust3r:.1f}px")
        return fx_dust3r
    return None


# ---------------------------------------------------------------------------
# Isolation: Очистка от земли и шума
# ---------------------------------------------------------------------------

def isolate_pile(pts, colors):
    if not OPEN3D_AVAILABLE:
        return pts, colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # 1. RANSAC - Убираем землю
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.02,
        ransac_n=3,
        num_iterations=1000
    )
    pcd_no_ground = pcd.select_by_index(inliers, invert=True)
    
    # 2. DBSCAN
    pcd_down = pcd_no_ground.voxel_down_sample(voxel_size=0.005)
    labels = np.array(pcd_down.cluster_dbscan(eps=0.05, min_points=100))

    if labels.max() < 0:
        print("[DUSt3R_GPU] Warning: No valid clusters found after ground removal.")
        return pts, colors

    biggest = np.bincount(labels[labels >= 0]).argmax()
    
    # 3. BBox
    cluster_down_pcd = pcd_down.select_by_index(np.where(labels == biggest)[0])
    bbox = cluster_down_pcd.get_axis_aligned_bounding_box()
    bbox.scale(1.05, bbox.get_center()) 
    
    pile_pcd = pcd_no_ground.crop(bbox)
    print(f"[DUSt3R_GPU] Pile isolated: {len(pile_pcd.points)} points "
          f"({len(pile_pcd.points)/len(pts)*100:.1f}% of total)")

    pile_pts = np.asarray(pile_pcd.points)
    pile_colors = (np.asarray(pile_pcd.colors) * 255.0).astype(np.uint8)

    return pile_pts, pile_colors


# ---------------------------------------------------------------------------
# Volume: Пуассон с фоллбэком на ConvexHull
# ---------------------------------------------------------------------------

def compute_volume(pts):
    from scipy.spatial import ConvexHull
    
    # Отзеркаливаем кучу вниз для создания замкнутого объема
    z_min = pts[:, 2].min()
    mirrored = pts.copy()
    mirrored[:, 2] = 2 * z_min - mirrored[:, 2]
    full_pts = np.vstack([pts, mirrored])

    # -- ПОПЫТКА ЧЕРЕЗ МЕШ ПУАССОНА --
    if OPEN3D_AVAILABLE:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(full_pts)
            
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(k=15)
            
            # Строим меш. Не обрезаем края по плотности, чтобы он остался watertight
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            
            if mesh.is_watertight():
                vol = mesh.get_volume() / 2.0  # Делим пополам из-за отзеркаливания
                print(f"[DUSt3R_GPU] Volume calculated using Poisson Mesh: {vol:.6f} units")
                return vol
            else:
                print("[DUSt3R_GPU] Poisson mesh is not watertight. Falling back to ConvexHull.")
        except Exception as e:
            print(f"[DUSt3R_GPU] Poisson volume error: {e}. Falling back to ConvexHull.")

    # -- РЕЗЕРВНЫЙ ВАРИАНТ ЧЕРЕЗ CONVEX HULL --
    hull = ConvexHull(full_pts)
    vol = hull.volume / 2.0
    print(f"[DUSt3R_GPU] Volume calculated using ConvexHull: {vol:.6f} units")
    return vol


# ---------------------------------------------------------------------------
# PLY save
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# GLB mesh (Визуальный меш для вывода)
# ---------------------------------------------------------------------------

def pointcloud_to_mesh(pts, colors, output_glb_path):
    if not OPEN3D_AVAILABLE:
        return None

    print(f"[DUSt3R_GPU] Creating visual mesh from {len(pts)} points using Poisson...")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)

    bbox = np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
    voxel_size = float(np.min(bbox)) * 0.005 
    voxel_size = max(0.001, min(voxel_size, 0.02)) 
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    normal_radius = voxel_size * 10
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=normal_radius, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)

    # Удаляем артефакты (визуальный меш перестает быть watertight, но выглядит красиво)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.15)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    mesh_colors = []
    for v in np.asarray(mesh.vertices):
        _, idx, _ = kdtree.search_knn_vector_3d(v, 1)
        mesh_colors.append(np.asarray(pcd.colors)[idx[0]])
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack(mesh_colors))

    R = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(R, center=mesh.get_center())

    mesh_colors_np = np.power(np.asarray(mesh.vertex_colors), 1.5)
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors_np)

    o3d.io.write_triangle_mesh(output_glb_path, mesh, write_ascii=False, compressed=True)
    print(f"[DUSt3R_GPU] Visual mesh saved to {output_glb_path} ({len(mesh.triangles)} faces)")
    return output_glb_path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",  required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--exif_json",  default=None)
    parser.add_argument("--no_glb",     action="store_true")
    parser.add_argument("--cube_square_m", type=float, default=0.02)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[DUSt3R_GPU] Device: {device.upper()}")

    ckpt_paths = glob("checkpoints/*.pth")
    if not ckpt_paths:
        raise FileNotFoundError("No checkpoint in checkpoints/")
    model = AsymmetricCroCo3DStereo.from_pretrained(ckpt_paths[0]).to(device)

    img_files = sorted(
        glob(os.path.join(args.image_dir, "*.jpg")) +
        glob(os.path.join(args.image_dir, "*.png"))
    )
    if len(img_files) < 2:
        raise ValueError(f"Need at least 2 images, found {len(img_files)}")
    imgs = load_images(img_files, size=512)

    gps_coords  = None
    known_focal = None
    if args.exif_json and os.path.exists(args.exif_json):
        with open(args.exif_json) as f:
            exif_list = json.load(f)
        known_focal = focal_px_from_exif(exif_list, dust3r_img_size=512)
        sorted_exif = sorted(exif_list, key=lambda e: e.get("photo_index", 0))
        gps_list = []
        for entry in sorted_exif:
            if not isinstance(entry, dict): continue
            inner = entry.get("exif", entry)
            if not isinstance(inner, dict): inner = entry
            if not isinstance(inner, dict): continue
            lat = inner.get("lat")
            lon = inner.get("lon")
            if lat is not None and lon is not None:
                gps_list.append((lat, lon))
        if len(gps_list) == len(img_files):
            gps_coords = gps_list

    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    print("[DUSt3R_GPU] Running inference...")
    output = inference(pairs, model, device, batch_size=2)

    print("[DUSt3R_GPU] Global Alignment...")
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=0.01)

    pts3d    = scene.get_pts3d()
    masks    = scene.get_masks()
    rgb_imgs = scene.imgs

    pts3d_np = [p.detach().cpu().numpy() for p in pts3d]
    masks_np = [m.detach().cpu().numpy() for m in masks]

    all_pts, all_colors = [], []
    for i in range(len(pts3d_np)):
        mask = masks_np[i]
        all_pts.append(pts3d_np[i][mask])
        c = (rgb_imgs[i][mask] * 255.0).clip(0, 255).astype(np.uint8)
        all_colors.append(c)

    final_pts    = np.concatenate(all_pts,    axis=0)
    final_colors = np.concatenate(all_colors, axis=0)

    # -- ПОИСК МАСШТАБА --
    scale_cube = compute_scale_from_cube(
        img_files, pts3d_np, masks_np,
        square_size_m=args.cube_square_m, pattern=(3, 3), dust3r_size=512
    )

    scale_weighted, scale_median = None, None
    if gps_coords is not None:
        poses       = scene.get_im_poses().detach().cpu().numpy()
        cam_centers = poses[:, :3, 3]
        scale_weighted, scale_median = compute_scale_from_gps(gps_coords, cam_centers)

    # Игнорируем GPS, если куб успешно найден (чтобы избежать огромной погрешности)
    if scale_cube is not None:
        print("[DUSt3R_GPU] Accurate cube scale found. Discarding GPS scales to prevent discrepancy.")
        scale_weighted = None
        scale_median   = None

    best_scale = scale_cube if scale_cube is not None else scale_weighted
    scale_source = "cube" if scale_cube is not None else ("gps_weighted" if scale_weighted else None)

    # -- ИЗОЛЯЦИЯ КУЧИ --
    print("[DUSt3R_GPU] Isolating target material from scene...")
    clean_pts, clean_colors = isolate_pile(final_pts, final_colors)
    if len(clean_pts) > 100:
        final_pts = clean_pts
        final_colors = clean_colors

    # -- РАСЧЕТ ОБЪЕМА --
    volume_units  = None
    volume_m3     = None
    volume_m3_med = None

    try:
        # Теперь считаем объем по обновленной логике (Mesh Poisson / ConvexHull)
        volume_units = compute_volume(final_pts)

        if best_scale is not None:
            volume_m3 = volume_units * (best_scale ** 3)
            print(f"[DUSt3R_GPU] Volume ({scale_source}): {volume_m3:.3f} m3")

        if scale_median is not None:
            volume_m3_med = volume_units * (scale_median ** 3)

    except ImportError:
        print("[DUSt3R_GPU] scipy missing: pip install scipy")
    except Exception as e:
        print(f"[DUSt3R_GPU] Volume error: {e}")

    # Сохраняем в PLY только очищенную кучу
    save_ply(args.output, final_pts, final_colors)

    ply_base64 = None
    try:
        with open(args.output, "rb") as f:
            ply_base64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        pass

    glb_base64    = None
    glb_file_path = None

    if not args.no_glb and OPEN3D_AVAILABLE and len(final_pts) > 1000:
        try:
            glb_file_path = args.output.replace(".ply", ".glb")
            pointcloud_to_mesh(final_pts, final_colors, glb_file_path)
            with open(glb_file_path, "rb") as f:
                glb_base64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"[DUSt3R_GPU] GLB generation error: {e}")

    result = {
        "status":              "success",
        "point_count":         len(final_pts),
        "scale_cube":          round(scale_cube,     4) if scale_cube     is not None else None,
        "scale_weighted":      round(scale_weighted, 4) if scale_weighted is not None else None,
        "scale_median":        round(scale_median,   4) if scale_median   is not None else None,
        "scale_source":        scale_source,
        "volume_dust3r_units": round(volume_units,   6) if volume_units   is not None else None,
        "volume_m3":           round(volume_m3,      4) if volume_m3      is not None else None,
        "volume_m3_weighted":  round(volume_m3,      3) if (volume_m3 is not None and scale_source != "cube") else None,
        "volume_m3_median":    round(volume_m3_med,  3) if volume_m3_med  is not None else None,
        "ply_file":            args.output,
        "ply_base64":          ply_base64,
        "glb_file":            glb_file_path,
        "glb_base64":          glb_base64,
    }
    result_path = args.output.replace(".ply", "_result.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    log_result = {k: v for k, v in result.items() if k not in ["ply_base64", "glb_base64"]}
    print(json.dumps(log_result))
    print("[DUSt3R_GPU] Done!")

if __name__ == '__main__':
    main()