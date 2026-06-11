import asyncio
import json
import os
import subprocess
import uvicorn
import httpx
from fastapi import FastAPI, HTTPException, Request
from typing import Optional

app = FastAPI(title="Volmetric DUSt3R Receiver")

SUPABASE_URL = "https://supabase.gottland.ru"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoic2VydmljZV9yb2xlIiwiaXNzIjoic3VwYWJhc2UiLCJpYXQiOjE3NzU5NDEyMDAsImV4cCI6MTkzMzcwNzYwMH0.PRYMYV77jrsZT7sXewMus7EO5eISiXuOdeth--yHNHQ"

HEADERS_JSON = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
    "Accept":        "application/json",
}

PLY_BUCKET = "dust3r-ply"


# ---------------------------------------------------------------------------
# Скачивание фото из Supabase
# ---------------------------------------------------------------------------

async def download_image(client: httpx.AsyncClient, photo_id: str, save_path: str):
    db_url = f"{SUPABASE_URL}/rest/v1/colmap_photos?id=eq.{photo_id}&select=public_url"
    db_resp = await client.get(db_url, headers=HEADERS_JSON, timeout=10.0)

    if db_resp.status_code != 200 or not db_resp.json():
        raise Exception(f"Photo {photo_id} not found (status {db_resp.status_code})")

    image_url = db_resp.json()[0].get("public_url")
    if not image_url:
        raise Exception(f"public_url is empty for photo {photo_id}")

    img_resp = await client.get(image_url, timeout=30.0)
    if img_resp.status_code != 200:
        raise Exception(f"Cannot download {image_url} (status {img_resp.status_code})")

    with open(save_path, "wb") as f:
        f.write(img_resp.content)
    print(f"  -> {save_path} downloaded")


# ---------------------------------------------------------------------------
# Загрузка файла в Supabase Storage
# ---------------------------------------------------------------------------

async def upload_to_supabase_storage(analysis_id: str, file_path: str,
                                      filename: str, content_type: str) -> Optional[str]:
    storage_path = f"{analysis_id}/{filename}"
    upload_url   = f"{SUPABASE_URL}/storage/v1/object/{PLY_BUCKET}/{storage_path}"
    public_url   = f"{SUPABASE_URL}/storage/v1/object/public/{PLY_BUCKET}/{storage_path}"

    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        print(f"[DUSt3R] Uploading {filename} ({len(file_bytes)//1024} KB)...")

        async with httpx.AsyncClient() as client:
            upload_resp = await client.post(
                upload_url,
                content=file_bytes,
                headers={
                    "apikey":        SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type":  content_type,
                    "x-upsert":      "true",
                },
                timeout=120.0,
            )

            if upload_resp.status_code not in (200, 201):
                print(f"[DUSt3R] {filename} upload failed: "
                      f"{upload_resp.status_code} {upload_resp.text}")
                return None

            print(f"[DUSt3R] {filename} uploaded -> {public_url}")
            return public_url

    except Exception as e:
        print(f"[DUSt3R] {filename} upload error: {e}")
        return None


# ---------------------------------------------------------------------------
# Запись результатов в БД
# ---------------------------------------------------------------------------

async def save_results_to_db(analysis_id: str, ply_url: Optional[str],
                              glb_url: Optional[str], gpu_result: dict):
    """
    Сохраняет ссылки на файлы и метрики в таблицу dust3r_results.
    """
    payload = {"analysis_id": analysis_id}
    if ply_url:
        payload["ply_url"] = ply_url
    if glb_url:
        payload["glb_url"] = glb_url

    # Метрики объёма и масштаба
    for key in ("scale_cube", "scale_weighted", "scale_median", "scale_source",
                "volume_dust3r_units", "volume_m3", "volume_m3_weighted", "volume_m3_median",
                "point_count"):
        val = gpu_result.get(key)
        if val is not None:
            payload[key] = val

    try:
        async with httpx.AsyncClient() as client:
            db_resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/dust3r_results",
                headers=HEADERS_JSON,
                json=payload,
                timeout=10.0,
            )
            if db_resp.status_code in (200, 201):
                print(f"[DUSt3R] DB record saved for analysis {analysis_id}")
            else:
                print(f"[DUSt3R] DB write warning: {db_resp.status_code} {db_resp.text}")
    except Exception as e:
        print(f"[DUSt3R] DB write error: {e}")


# ---------------------------------------------------------------------------
# Инференс (блокирующий → запускаем в потоке)
# ---------------------------------------------------------------------------

def _run_inference(project_dir: str, output_ply: str, exif_json_path: Optional[str]):
    cmd = ["python", "dust3r_gpu.py", "--image_dir", project_dir, "--output", output_ply]
    if exif_json_path and os.path.exists(exif_json_path):
        cmd += ["--exif_json", exif_json_path]
    subprocess.run(cmd, check=True, capture_output=False)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/run")
async def trigger_dust3r(request: Request):
    raw_body = await request.body()
    print(f"[DUSt3R] Request {len(raw_body)} bytes")

    try:
        body = json.loads(raw_body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    analysis_id = body.get("analysis_id")
    photo_ids   = body.get("photo_ids", [])
    exif_raw    = body.get("exif")

    print(f"[DUSt3R] analysis_id={analysis_id} | photos={len(photo_ids)} | "
          f"exif={'YES (' + str(len(exif_raw)) + ')' if exif_raw else 'NO'}")

    if not analysis_id or not photo_ids:
        raise HTTPException(status_code=400, detail="analysis_id and photo_ids are required")

    project_dir = f"./volmetric/projects/{analysis_id}"
    os.makedirs(project_dir, exist_ok=True)

    # --- Сохраняем EXIF ---
    exif_json_path = None
    if exif_raw:
        exif_json_path = os.path.join(project_dir, "exif.json")
        with open(exif_json_path, "w") as f:
            json.dump(exif_raw, f)

        first_entry = exif_raw[0] if exif_raw else None
        first = None
        if isinstance(first_entry, dict):
            first = first_entry.get("exif") or first_entry
            if not isinstance(first, dict):
                first = None

        if first:
            print(f"[DUSt3R] EXIF saved: focal_real={first.get('focal_real')}mm "
                  f"orig={first.get('orig_width')}x{first.get('orig_height')}")
        else:
            print(f"[DUSt3R] EXIF saved: {len(exif_raw)} entries, no focal/GPS metadata")

    # --- Скачиваем фото параллельно ---
    print("[DUSt3R] Downloading photos in parallel...")
    try:
        async with httpx.AsyncClient() as client:
            tasks = [
                download_image(client, photo_id, os.path.join(project_dir, f"{idx:03d}.jpg"))
                for idx, photo_id in enumerate(photo_ids)
            ]
            await asyncio.gather(*tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Photo download error: {e}")

    # --- Инференс ---
    output_ply = os.path.join(project_dir, "dust3r_output.ply")
    print("[DUSt3R] Running inference...")
    try:
        await asyncio.to_thread(_run_inference, project_dir, output_ply, exif_json_path)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # --- Читаем результат из JSON ---
    result_json_path = output_ply.replace(".ply", "_result.json")
    gpu_result = {}
    if os.path.exists(result_json_path):
        with open(result_json_path) as f:
            gpu_result = json.load(f)

    # --- Загружаем PLY в Supabase Storage ---
    ply_url = None
    if os.path.exists(output_ply):
        ply_url = await upload_to_supabase_storage(
            analysis_id, output_ply, "dust3r_output.ply", "application/octet-stream"
        )

    # --- Загружаем GLB в Supabase Storage ---
    output_glb = output_ply.replace(".ply", ".glb")
    glb_url = None
    if os.path.exists(output_glb):
        glb_url = await upload_to_supabase_storage(
            analysis_id, output_glb, "dust3r_output.glb", "model/gltf-binary"
        )

    # --- Сохраняем в БД ---
    if ply_url or glb_url:
        await save_results_to_db(analysis_id, ply_url, glb_url, gpu_result)

    # --- Ответ для n8n ---
    result = {
        "status":              "success",
        "analysis_id":         analysis_id,
        "point_count":         gpu_result.get("point_count"),

        # Масштабы
        "scale_cube":          gpu_result.get("scale_cube"),
        "scale_source":        gpu_result.get("scale_source"),
        "scale_weighted":      gpu_result.get("scale_weighted"),
        "scale_median":        gpu_result.get("scale_median"),

        # Объёмы
        "volume_dust3r_units": gpu_result.get("volume_dust3r_units"),
        "volume_m3":           gpu_result.get("volume_m3"),        # лучший scale (куб или GPS)
        "volume_m3_weighted":  gpu_result.get("volume_m3_weighted"),
        "volume_m3_median":    gpu_result.get("volume_m3_median"),

        # Файлы
        "ply_url":             ply_url,
        "ply_file":            output_ply,
        "glb_url":             glb_url,
        "glb_file":            output_glb,
    }

    # Лог
    v   = result.get("volume_m3")
    src = result.get("scale_source", "?")
    if v is not None:
        print(f"[DUSt3R] Volume ({src}): {v} m3")
    if ply_url:
        print(f"[DUSt3R] PLY URL: {ply_url}")
    if glb_url:
        print(f"[DUSt3R] GLB URL: {glb_url}")

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006)