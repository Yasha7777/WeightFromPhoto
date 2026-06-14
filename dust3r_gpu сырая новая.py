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
# Cube scale
# ---------------------------------------------------------------------------

def compute_scale_from_cube(img_files, pts3d_np, masks_np,
                             square_size_m=0.01775,
                             pattern=(3, 3),
                             dust3r_size=512):
    """
    Ищет шахматный паттерн куба на фото, вычисляет масштаб сцены.
    Возвращает (scale_cube, cube_3d_centers) где cube_3d_centers — список
    3D-центров найденного куба (для последующего исключения из облака точек).
    """
    if not CV2_AVAILABLE:
        print("[DUSt3R_GPU] OpenCV not available, skipping cube scale")
        return None, []

    all_scales = []
    cube_3d_centers = []  # 3D-центры куба на каждом фото где он найден

    for img_idx, img_path in enumerate(img_files):
        img = cv2.imread(img_path)
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        def find_chess_in_crops(gray_img, pat):
            h, w = gray_img.shape
            crops = [
                (gray_img,                     0,    0   ),
                (gray_img[:h//2, :w//2],       0,    0   ),
                (gray_img[:h//2, w//2:],       0,    w//2),
                (gray_img[h//2:, :w//2],       h//2, 0   ),
                (gray_img[h//2:, w//2:],       h//2, w//2),
                (gray_img[:h//2, w//4:3*w//4], 0,    w//4),
                (gray_img[h//2:, w//4:3*w//4], h//2, w//4),
            ]
            flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                     cv2.CALIB_CB_NORMALIZE_IMAGE +
                     cv2.CALIB_CB_FAST_CHECK)
            for crop, oy, ox in crops:
                r, c = cv2.findChessboardCorners(crop, pat, flags)
                if r:
                    c[:, 0, 0] += ox
                    c[:, 0, 1] += oy
                    print(f"[DUSt3R_GPU] Cube: pattern found in crop offset ({ox},{oy})")
                    return True, c
            return False, None

        ret, corners = find_chess_in_crops(gray, pattern)
        if not ret:
            small = cv2.resize(gray, (orig_w // 2, orig_h // 2))
            ret, corners_small = find_chess_in_crops(small, pattern)
            if ret:
                corners = corners_small * 2.0
            else:
                print(f"[DUSt3R_GPU] Cube: no pattern on photo {img_idx}")
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

        # Сохраняем 3D-центр куба для этого фото
        valid_pts = [pts3d_corners[i] for i in range(len(pts3d_corners)) if valid_flags[i]]
        if valid_pts:
            cube_center_3d = np.mean(valid_pts, axis=0)
            cube_3d_centers.append(cube_center_3d)

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

        if img_scales:
            img_scales_arr = np.array(img_scales)
            img_median = float(np.median(img_scales_arr))
            valid_scales = img_scales_arr[
                (img_scales_arr >= 0.5 * img_median) &
                (img_scales_arr <= 2.0 * img_median)
            ]
            if len(valid_scales) > 0:
                img_scale = float(np.median(valid_scales))

                # --- SANITY CHECK ---
                # Съёмка объекта с расстояния ~30см–2м.
                # Если scale выходит за эти пределы — куб найден ошибочно
                # (например, шахматный паттерн на фоне или артефакт DUSt3R).
                SCALE_MIN = 0.05  # 5 см/unit — меньше не бывает при нормальной съёмке
                SCALE_MAX = 2.0   # 2 м/unit  — больше не бывает при съёмке объекта вблизи
                if not (SCALE_MIN <= img_scale <= SCALE_MAX):
                    print(f"[DUSt3R_GPU] Cube photo {img_idx}: scale={img_scale:.4f} REJECTED "
                          f"(out of plausible range {SCALE_MIN}–{SCALE_MAX} m/unit). "
                          f"Likely false detection or DUSt3R scale degenerate.")
                    continue

                all_scales.append(img_scale)
                print(f"[DUSt3R_GPU] Cube found on photo {img_idx}: scale={img_scale:.4f} m/unit "
                      f"(filtered {len(valid_scales)}/{len(img_scales)} valid pairs)")

    if not all_scales:
        print("[DUSt3R_GPU] Cube not found on any photo")
        return None, []

    scale_cube = float(np.median(all_scales))
    print(f"[DUSt3R_GPU] Cube scale FINAL: {scale_cube:.4f} m/unit")
    return scale_cube, cube_3d_centers


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
# Isolation: отделяем целевой объект, явно исключая куб
# ---------------------------------------------------------------------------

def isolate_target_object(pts, colors, scale_m_per_unit=None, cube_3d_centers=None):
    """
    Изолирует ЦЕЛЕВОЙ объект (кучу / банку) из облака точек.

    Логика:
    1. RANSAC убирает пол
    2. DBSCAN даёт кластеры
    3. Если известны 3D-центры куба → исключаем кластеры близкие к кубу
    4. Из оставшихся берём САМЫЙ БОЛЬШОЙ кластер (это и есть целевой объект)
    5. BBox вокруг него с небольшим запасом

    Параметры:
        cube_3d_centers: список np.array[3] — центры куба в 3D (из compute_scale_from_cube)
    """
    if not OPEN3D_AVAILABLE:
        return pts, colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # --- Параметры в единицах сцены ---
    # Sanity check: scale должен быть в разумном диапазоне.
    # Если нет — используем фиксированные эвристические параметры
    # (они работают хорошо при типичных DUSt3R-сценах съёмки вблизи).
    SCALE_REASONABLE_MIN = 0.05
    SCALE_REASONABLE_MAX = 2.0
    scale_ok = (scale_m_per_unit is not None and
                SCALE_REASONABLE_MIN <= scale_m_per_unit <= SCALE_REASONABLE_MAX)

    if scale_ok:
        ransac_dist           = 0.03  / scale_m_per_unit  # 3 см
        voxel_units           = 0.005 / scale_m_per_unit  # 5 мм
        eps_units             = 0.025 / scale_m_per_unit  # 2.5 см
        cube_exclusion_radius = 0.10  / scale_m_per_unit  # 10 см вокруг куба
        print(f"[DUSt3R_GPU] isolate_target: scale={scale_m_per_unit:.4f} "
              f"=> eps={eps_units:.5f} voxel={voxel_units:.5f} ransac={ransac_dist:.5f} "
              f"cube_excl_r={cube_exclusion_radius:.5f}")
    else:
        # Фиксированные параметры — хорошо работают для сцен ~0.3–1.5м
        ransac_dist           = 0.02
        voxel_units           = 0.005
        eps_units             = 0.05
        cube_exclusion_radius = 0.15
        if scale_m_per_unit is not None:
            print(f"[DUSt3R_GPU] isolate_target: scale={scale_m_per_unit:.4f} is UNREALISTIC, "
                  f"using fixed params (eps={eps_units}, voxel={voxel_units}, ransac={ransac_dist})")
        else:
            print(f"[DUSt3R_GPU] isolate_target: no scale, "
                  f"using fixed params (eps={eps_units}, voxel={voxel_units}, ransac={ransac_dist})")

    # 1. Убираем пол (RANSAC)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=ransac_dist,
        ransac_n=3,
        num_iterations=1000
    )
    pcd_no_ground = pcd.select_by_index(inliers, invert=True)
    print(f"[DUSt3R_GPU] After ground removal: {len(pcd_no_ground.points)} points")

    # 2. Даунсемплинг + DBSCAN
    pcd_down = pcd_no_ground.voxel_down_sample(voxel_size=voxel_units)
    labels   = np.array(pcd_down.cluster_dbscan(eps=eps_units, min_points=50))
    pcd_down_pts = np.asarray(pcd_down.points)

    n_clusters = labels.max() + 1
    if n_clusters <= 0:
        print("[DUSt3R_GPU] Warning: No clusters found after ground removal. Returning all points.")
        return pts, colors

    print(f"[DUSt3R_GPU] DBSCAN found {n_clusters} clusters (+ noise)")

    # 3. Собираем информацию по каждому кластеру
    cluster_info = {}
    for label_id in range(n_clusters):
        mask = labels == label_id
        count = mask.sum()
        if count < 50:
            continue
        cluster_pts    = pcd_down_pts[mask]
        cluster_center = cluster_pts.mean(axis=0)

        # Проверяем: является ли этот кластер кубом?
        is_cube = False
        if cube_3d_centers:
            for cc in cube_3d_centers:
                dist = np.linalg.norm(cluster_center - np.array(cc))
                if dist < cube_exclusion_radius:
                    is_cube = True
                    break

        cluster_info[label_id] = {
            'count':   count,
            'center':  cluster_center,
            'is_cube': is_cube,
        }
        marker = " [CUBE - excluded]" if is_cube else ""
        print(f"[DUSt3R_GPU]   Cluster #{label_id}: {count} pts, "
              f"center={cluster_center.round(4)}{marker}")

    # 4. Фильтруем: убираем кластеры-кубы
    target_candidates = {k: v for k, v in cluster_info.items() if not v['is_cube']}

    if not target_candidates:
        print("[DUSt3R_GPU] Warning: All clusters excluded as cube. Returning non-ground points.")
        pile_pts    = np.asarray(pcd_no_ground.points)
        pile_colors = (np.asarray(pcd_no_ground.colors) * 255.0).astype(np.uint8)
        return pile_pts, pile_colors

    # 5. Целевой объект = самый большой оставшийся кластер
    target_id = max(target_candidates.keys(), key=lambda k: target_candidates[k]['count'])
    target_info = target_candidates[target_id]
    print(f"[DUSt3R_GPU] Target cluster: #{target_id} with {target_info['count']} pts "
          f"(largest non-cube cluster)")

    # 6. BBox вокруг целевого кластера (с запасом 5%)
    target_mask = labels == target_id
    target_down_pcd = pcd_down.select_by_index(np.where(target_mask)[0])

    bbox = target_down_pcd.get_axis_aligned_bounding_box()
    bbox.scale(1.05, bbox.get_center())

    pile_pcd = pcd_no_ground.crop(bbox)
    print(f"[DUSt3R_GPU] Target isolated: {len(pile_pcd.points)} points "
          f"({len(pile_pcd.points)/len(pts)*100:.1f}% of total)")

    pile_pts    = np.asarray(pile_pcd.points)
    pile_colors = (np.asarray(pile_pcd.colors) * 255.0).astype(np.uint8)
    return pile_pts, pile_colors


# ---------------------------------------------------------------------------
# Volume: Poisson с фоллбэком на ConvexHull
# ---------------------------------------------------------------------------

def compute_volume(pts):
    """
    Считает объём облака точек.

    Метод: зеркалируем объект вниз (создаём замкнутую нижнюю границу),
    строим Poisson mesh, считаем его объём / 2.
    Если Poisson не дал watertight — падаем на ConvexHull.
    """
    from scipy.spatial import ConvexHull

    # Зеркалируем по дну объекта (создаём замкнутость снизу)
    z_min = pts[:, 2].min()
    mirrored = pts.copy()
    mirrored[:, 2] = 2 * z_min - mirrored[:, 2]
    full_pts = np.vstack([pts, mirrored])

    if OPEN3D_AVAILABLE:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(full_pts)

            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

            if mesh.is_watertight():
                vol = mesh.get_volume() / 2.0
                print(f"[DUSt3R_GPU] Volume calculated using Poisson Mesh: {vol:.6f} units")
                return vol
            else:
                print("[DUSt3R_GPU] Poisson mesh is not watertight. Falling back to ConvexHull.")
        except Exception as e:
            print(f"[DUSt3R_GPU] Poisson volume error: {e}. Falling back to ConvexHull.")

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
# GLB mesh (визуальный меш для отображения)
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
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)

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
    parser.add_argument("--image_dir",     required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--exif_json",     default=None)
    parser.add_argument("--no_glb",        action="store_true")
    parser.add_argument("--cube_square_m", type=float, default=0.01775,
                        help="Размер одной клетки куба в метрах (7.1см / 4 = 0.01775)")
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

    known_focal = None
    if args.exif_json and os.path.exists(args.exif_json):
        with open(args.exif_json) as f:
            exif_list = json.load(f)
        known_focal = focal_px_from_exif(exif_list, dust3r_img_size=512)

    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    print("[DUSt3R_GPU] Running inference...")
    output = inference(pairs, model, device, batch_size=2)

    print("[DUSt3R_GPU] Global Alignment...")
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)

    # Фиксируем focal length из EXIF — не даём DUSt3R его "угадывать".
    # Без этого при малотекстурных объектах (бетон, однородные поверхности)
    # оптимизатор искажает масштаб сцены в разы.
    if known_focal is not None:
        print(f"[DUSt3R_GPU] Fixing focal to EXIF value: {known_focal:.1f}px (freezing im_focals)")
        try:
            import torch as _torch
            for i in range(len(imgs)):
                scene.im_focals[i].data.fill_(known_focal)
                scene.im_focals[i].requires_grad_(False)
            print("[DUSt3R_GPU] Optimizing: pw_poses / im_depthmaps / im_poses (focal FIXED)")
        except Exception as e:
            print(f"[DUSt3R_GPU] Could not fix focal: {e}")
    else:
        print("[DUSt3R_GPU] No EXIF focal — focal will be optimized freely")

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

    # -- ПОИСК МАСШТАБА ПО КУБУ (теперь возвращает и 3D-центры куба) --
    scale_cube, cube_3d_centers = compute_scale_from_cube(
        img_files, pts3d_np, masks_np,
        square_size_m=args.cube_square_m,
        pattern=(3, 3),
        dust3r_size=512
    )




    # Финальный sanity check на best_scale
    SCALE_MIN, SCALE_MAX = 0.05, 2.0
    if best_scale is not None and not (SCALE_MIN <= best_scale <= SCALE_MAX):
        print(f"[DUSt3R_GPU] WARNING: best_scale={best_scale:.4f} is outside "
              f"plausible range [{SCALE_MIN}, {SCALE_MAX}]. "
              f"Volume in m3 will NOT be calculated to avoid garbage output.")
        best_scale   = None
        scale_source = None

    # -- ИЗОЛЯЦИЯ ЦЕЛЕВОГО ОБЪЕКТА (куб явно исключается) --
    print("[DUSt3R_GPU] Isolating target object from scene...")
    clean_pts, clean_colors = isolate_target_object(
        final_pts, final_colors,
        scale_m_per_unit=best_scale,
        cube_3d_centers=cube_3d_centers if cube_3d_centers else None
    )
    if len(clean_pts) > 100:
        final_pts    = clean_pts
        final_colors = clean_colors

    # -- РАСЧЁТ ОБЪЁМА --
    volume_units  = None
    volume_m3     = None

    try:
        volume_units = compute_volume(final_pts)

        if best_scale is not None:
            volume_m3 = volume_units * (best_scale ** 3)
            print(f"[DUSt3R_GPU] Volume ({scale_source}): {volume_m3:.4f} m3 "
                  f"= {volume_m3 * 1000:.2f} litres")


    except ImportError:
        print("[DUSt3R_GPU] scipy missing: pip install scipy")
    except Exception as e:
        print(f"[DUSt3R_GPU] Volume error: {e}")

    save_ply(args.output, final_pts, final_colors)

    ply_base64 = None
    try:
        with open(args.output, "rb") as f:
            ply_base64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
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
        "scale_source":        scale_source,
        "volume_dust3r_units": round(volume_units,   6) if volume_units   is not None else None,
        "volume_m3":           round(volume_m3,      4) if volume_m3      is not None else None,
        "volume_litres":       round(volume_m3 * 1000, 2) if volume_m3    is not None else None,
        "cube_found_on_n_photos": len(cube_3d_centers),
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