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

# Название бакета в Supabase Storage
# Создайте его вручную в Supabase Dashboard → Storage → New bucket → "dust3r-ply" (public)
PLY_BUCKET = "dust3r-ply"


# ---------------------------------------------------------------------------
# Скачивание фото из Supabase
# ---------------------------------------------------------------------------

async def download_image(photo_id: str, save_path: str):
    async with httpx.AsyncClient() as client:
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


# ---------------------------------------------------------------------------
# Загрузка PLY в Supabase Storage + запись ссылки в таблицу
# ---------------------------------------------------------------------------

async def upload_ply_to_supabase(analysis_id: str, ply_path: str) -> Optional[str]:
    """
    Загружает .ply файл в бакет dust3r-ply и возвращает публичную ссылку.
    Также записывает ссылку в таблицу dust3r_results (создайте её заранее — см. ниже).

    SQL для создания таблицы:
        CREATE TABLE dust3r_results (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            analysis_id UUID NOT NULL,
            ply_url     TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );

    Бакет: Supabase Dashboard → Storage → New bucket → Name: "dust3r-ply", Public: ON
    """
    storage_path = f"{analysis_id}/dust3r_output.ply"
    upload_url   = f"{SUPABASE_URL}/storage/v1/object/{PLY_BUCKET}/{storage_path}"
    public_url   = f"{SUPABASE_URL}/storage/v1/object/public/{PLY_BUCKET}/{storage_path}"

    try:
        with open(ply_path, "rb") as f:
            ply_bytes = f.read()

        print(f"[DUSt3R] Uploading PLY to Supabase Storage "
              f"({len(ply_bytes)//1024//1024} MB)...")

        async with httpx.AsyncClient() as client:
            # 1. Загружаем файл
            upload_resp = await client.post(
                upload_url,
                content=ply_bytes,
                headers={
                    "apikey":          SUPABASE_KEY,
                    "Authorization":   f"Bearer {SUPABASE_KEY}",
                    "Content-Type":    "application/octet-stream",
                    "x-upsert":        "true",   # перезаписать если уже есть
                },
                timeout=120.0,
            )

            if upload_resp.status_code not in (200, 201):
                print(f"[DUSt3R] PLY upload failed: {upload_resp.status_code} {upload_resp.text}")
                return None

            print(f"[DUSt3R] PLY uploaded -> {public_url}")

            # 2. Записываем ссылку в таблицу dust3r_results
            db_resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/dust3r_results",
                headers=HEADERS_JSON,
                json={"analysis_id": analysis_id, "ply_url": public_url},
                timeout=10.0,
            )

            if db_resp.status_code in (200, 201):
                print(f"[DUSt3R] DB record saved for analysis {analysis_id}")
            else:
                print(f"[DUSt3R] DB write warning: {db_resp.status_code} {db_resp.text}")

        return public_url

    except Exception as e:
        print(f"[DUSt3R] PLY upload error: {e}")
        return None


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
        first = exif_raw[0].get("exif", exif_raw[0])
        print(f"[DUSt3R] EXIF saved: focal_real={first.get('focal_real')}mm "
              f"orig={first.get('orig_width')}x{first.get('orig_height')}")

    # --- Скачиваем фото ---
    print("[DUSt3R] Downloading photos...")
    try:
        for idx, photo_id in enumerate(photo_ids):
            save_path = os.path.join(project_dir, f"{idx:03d}.jpg")
            await download_image(photo_id, save_path)
            print(f"  -> {save_path}")
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
        ply_url = await upload_ply_to_supabase(analysis_id, output_ply)

    # --- Формируем ответ для n8n ---
    result = {
        "status":              "success",
        "analysis_id":         analysis_id,
        "point_count":         gpu_result.get("point_count"),

        # Масштаб (обе версии, подробнее в документации)
        "scale_weighted":      gpu_result.get("scale_weighted"),   # топ-30% длинных GPS-пар
        "scale_median":        gpu_result.get("scale_median"),     # медиана всех GPS-пар

        # Объём в единицах DUSt3R (не зависит от GPS, стабилен)
        "volume_dust3r_units": gpu_result.get("volume_dust3r_units"),

        # Объём в м³ через weighted масштаб
        "volume_m3_weighted":  gpu_result.get("volume_m3_weighted"),

        # Объём в м³ через медианный масштаб
        "volume_m3_median":    gpu_result.get("volume_m3_median"),

        # Ссылка на PLY файл для просмотра на сайте
        "ply_url":             ply_url,
        "ply_file":            output_ply,
    }

    v_w = result.get("volume_m3_weighted")
    v_m = result.get("volume_m3_median")
    if v_w is not None:
        print(f"[DUSt3R] Volume weighted: {v_w} m3 | median: {v_m} m3")
        print(f"[DUSt3R] WARNING: GPS accuracy ~3-6m, volume error can be up to 4x")
    if ply_url:
        print(f"[DUSt3R] PLY URL: {ply_url}")

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006)
