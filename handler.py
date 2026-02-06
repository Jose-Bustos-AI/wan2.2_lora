import os
import json
import time
import uuid
import random
import logging
import urllib.parse

import requests
import websocket
import runpod

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("handler")

# -----------------------------------------------------------------------------
# Env
# -----------------------------------------------------------------------------
COMFY_HOST = os.getenv("COMFY_HOST", "127.0.0.1:8188")
COMFY_ROOT = os.getenv("COMFY_ROOT", "/comfyui")
WORKFLOW_PATH = os.getenv("WORKFLOW_PATH", "/app/workflows/workflow.json")

LORAS_DIR = os.path.join(COMFY_ROOT, "models", "loras")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "results")
SUPABASE_PATH_PREFIX = os.environ.get("SUPABASE_PATH_PREFIX", "runpod").strip("/")

# Which node receives LoRA injection (defaults to node "68" from your JSON: LoraLoader)
LORA_NODE_ID = os.getenv("LORA_NODE_ID", "68")

# -----------------------------------------------------------------------------
# Supabase upload
# -----------------------------------------------------------------------------
def supabase_upload_bytes(content: bytes, filename: str, content_type: str = "image/png") -> str:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

    path = f"{SUPABASE_PATH_PREFIX}/lora-image/{time.strftime('%Y/%m/%d')}/{filename}"
    upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"

    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": content_type,
        "x-upsert": "true",
    }

    r = requests.put(upload_url, headers=headers, params={"upsert": "true"}, data=content, timeout=180)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Supabase upload failed: {r.status_code} {r.text}")

    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{path}"

# -----------------------------------------------------------------------------
# Comfy helpers
# -----------------------------------------------------------------------------
def check_server(retries=240, delay=0.5) -> None:
    url = f"http://{COMFY_HOST}/"
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(delay)
    raise RuntimeError(f"ComfyUI not reachable at {url}")

def load_workflow() -> dict:
    if not os.path.exists(WORKFLOW_PATH):
        raise FileNotFoundError(f"Workflow not found: {WORKFLOW_PATH}")
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def queue_workflow(workflow: dict, client_id: str) -> str:
    payload = {"prompt": workflow, "client_id": client_id}
    r = requests.post(
        f"http://{COMFY_HOST}/prompt",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if r.status_code == 400:
        raise RuntimeError(f"ComfyUI /prompt 400: {r.text}")
    r.raise_for_status()
    data = r.json()
    pid = data.get("prompt_id")
    if not pid:
        raise RuntimeError(f"Missing prompt_id in response: {data}")
    return pid

def wait_for_done(ws: websocket.WebSocket, prompt_id: str, timeout_sec: int = 1800) -> None:
    start = time.time()
    while True:
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"ComfyUI execution timeout after {timeout_sec}s")

        out = ws.recv()
        if isinstance(out, str):
            msg = json.loads(out)
            t = msg.get("type")

            if t == "execution_error":
                data = msg.get("data", {})
                if data.get("prompt_id") == prompt_id:
                    raise RuntimeError(f"ComfyUI execution_error: {data}")

            if t == "executing":
                data = msg.get("data", {})
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    return

def get_history(prompt_id: str) -> dict:
    r = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30)
    r.raise_for_status()
    return r.json()

def get_image_bytes(filename: str, subfolder: str, image_type: str) -> bytes:
    qs = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder or "", "type": image_type or "output"})
    r = requests.get(f"http://{COMFY_HOST}/view?{qs}", timeout=60)
    r.raise_for_status()
    return r.content

# -----------------------------------------------------------------------------
# LoRA download
# -----------------------------------------------------------------------------
def ensure_lora(lora_url: str, lora_name: str) -> str:
    if not lora_url:
        raise ValueError("lora_url is missing")
    if not lora_name:
        lora_name = os.path.basename(lora_url.split("?")[0]).strip()

    if not lora_name.endswith(".safetensors"):
        raise ValueError("LoRA must be .safetensors (lora_name)")

    os.makedirs(LORAS_DIR, exist_ok=True)
    dest = os.path.join(LORAS_DIR, lora_name)

    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        logger.info(f"[lora] already exists: {dest}")
        return lora_name

    logger.info(f"[lora] downloading: {lora_url} -> {dest}")
    with requests.get(lora_url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    if not os.path.exists(dest) or os.path.getsize(dest) < 1024:
        raise RuntimeError("Downloaded LoRA invalid (too small)")
    logger.info(f"[lora] downloaded OK: {dest} ({os.path.getsize(dest)} bytes)")
    return lora_name

# -----------------------------------------------------------------------------
# Workflow patching (based on your JSON)
# - Positive/Negative: nodes 3,4 (CLIPTextEncode)
# - Size: node 5 (EmptyHunyuanLatentVideo) width/height
# - Samplers: nodes 35,36 (KSamplerAdvanced) steps/cfg
# - Seed: node 67 (Seed) -> seed
# - LoRA: node 68 (LoraLoader) -> lora_name + strengths (by default)
# -----------------------------------------------------------------------------
def pick_seed(seed_val):
    try:
        if seed_val is None:
            return random.randint(0, 2**31 - 1)
        s = int(seed_val)
        if s < 0:
            return random.randint(0, 2**31 - 1)
        return s
    except Exception:
        return random.randint(0, 2**31 - 1)

def patch_workflow(wf: dict, job_input: dict) -> dict:
    prompt = job_input.get("prompt")
    negative = job_input.get("negative_prompt", "")

    width = int(job_input.get("width", 1024))
    height = int(job_input.get("height", 1024))
    steps = int(job_input.get("steps", 24))
    cfg = float(job_input.get("cfg", 1.2))
    seed = pick_seed(job_input.get("seed"))

    # text
    if "3" in wf and isinstance(wf["3"], dict):
        wf["3"].setdefault("inputs", {})["text"] = prompt
    if "4" in wf and isinstance(wf["4"], dict):
        wf["4"].setdefault("inputs", {})["text"] = negative or ""

    # latent size
    if "5" in wf and isinstance(wf["5"], dict):
        wf["5"].setdefault("inputs", {})["width"] = width
        wf["5"].setdefault("inputs", {})["height"] = height

    # samplers
    for sid in ("35", "36"):
        if sid in wf and isinstance(wf[sid], dict):
            inputs = wf[sid].setdefault("inputs", {})
            inputs["steps"] = steps
            inputs["cfg"] = cfg

    # seed
    if "67" in wf and isinstance(wf["67"], dict):
        wf["67"].setdefault("inputs", {})["seed"] = seed

    # lora injection (optional)
    lora_url = job_input.get("lora_url")
    lora_name = job_input.get("lora_name")
    lora_strength_model = float(job_input.get("lora_strength_model", 1.0))
    lora_strength_clip = float(job_input.get("lora_strength_clip", 1.0))

    if lora_url or lora_name:
        if not lora_url:
            raise ValueError("If you pass lora_name you must also pass lora_url")
        lora_name = ensure_lora(lora_url, lora_name)

        node_id = str(job_input.get("lora_node_id") or LORA_NODE_ID)
        if node_id in wf and isinstance(wf[node_id], dict):
            inputs = wf[node_id].setdefault("inputs", {})
            # Works for LoraLoader
            inputs["lora_name"] = lora_name
            if "strength_model" in inputs:
                inputs["strength_model"] = lora_strength_model
            else:
                inputs["strength_model"] = lora_strength_model
            if "strength_clip" in inputs:
                inputs["strength_clip"] = lora_strength_clip
            else:
                inputs["strength_clip"] = lora_strength_clip
            logger.info(f"[lora] applied on node {node_id}: {lora_name} sm={lora_strength_model} sc={lora_strength_clip}")
        else:
            raise KeyError(f"LoRA node_id={node_id} not found in workflow")

    logger.info(f"[patch] width={width} height={height} steps={steps} cfg={cfg} seed={seed}")
    return wf

# -----------------------------------------------------------------------------
# MAIN HANDLER
# -----------------------------------------------------------------------------
def handler(job):
    job_input = job.get("input", {}) or {}
    logger.info(f"Received keys: {list(job_input.keys())}")

    # required
    prompt = job_input.get("prompt")
    if not prompt or not isinstance(prompt, str):
        return {"error": "Missing required field: prompt (string)"}

    try:
        check_server()
        wf = load_workflow()
        wf = patch_workflow(wf, job_input)

        client_id = str(uuid.uuid4())
        ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"

        ws = websocket.WebSocket()
        ws.connect(ws_url, timeout=10)

        prompt_id = queue_workflow(wf, client_id)
        logger.info(f"Queued prompt_id={prompt_id}")

        wait_for_done(ws, prompt_id, timeout_sec=int(job_input.get("timeout_sec", 1800)))
        ws.close()

        history = get_history(prompt_id)
        if prompt_id not in history:
            return {"error": f"Prompt ID {prompt_id} not in history"}

        outputs = history[prompt_id].get("outputs", {})
        for _, node_out in outputs.items():
            images = node_out.get("images")
            if not images:
                continue
            for img in images:
                if img.get("type") == "temp":
                    continue
                filename = img.get("filename")
                subfolder = img.get("subfolder", "")
                img_type = img.get("type", "output")
                if not filename:
                    continue

                png_bytes = get_image_bytes(filename, subfolder, img_type)
                out_name = f"{uuid.uuid4()}.png"
                image_url = supabase_upload_bytes(png_bytes, out_name, "image/png")

                return {
                    "image_url": image_url,
                    "prompt_id": prompt_id,
                    "seed_used": int(wf.get("67", {}).get("inputs", {}).get("seed", -1)),
                    "width": int(job_input.get("width", 1024)),
                    "height": int(job_input.get("height", 1024)),
                }

        return {"error": "No output images found in history.outputs"}

    except Exception as e:
        logger.exception("Handler error")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
