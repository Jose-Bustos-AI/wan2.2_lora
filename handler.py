import os
import json
import time
import uuid
import runpod
import requests
import random
from urllib.parse import urlparse
from supabase import create_client, Client
from websocket import WebSocket

# --- Bootstrap & Config ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "results")
SUPABASE_PATH_PREFIX = os.environ.get("SUPABASE_PATH_PREFIX", "runpod")
COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
WORKFLOW_PATH = os.environ.get("WORKFLOW_PATH", "/app/workflows/workflow.json")
LORA_NODE_ID = os.environ.get("LORA_NODE_ID") # Optional fixed ID

if not SUPABASE_URL or not SUPABASE_KEY:
    print("FATAL: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set.")
    # We exit here because the service is fundamentally misconfigured for this worker type
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
COMFY_API_URL = f"http://{COMFY_HOST}"
COMFY_WS_URL = f"ws://{COMFY_HOST}/ws"

# --- Helpers ---

def download_file(url, path):
    print(f"Downloading {url} to {path}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

def get_node_by_class(workflow, class_type_substr):
    for node_id, node in workflow.items():
        if class_type_substr.lower() in node.get("class_type", "").lower():
            return node_id, node
    return None, None

def get_nodes_by_class(workflow, class_type_substr):
    nodes = []
    for node_id, node in workflow.items():
        if class_type_substr.lower() in node.get("class_type", "").lower():
            nodes.append((node_id, node))
    return nodes

def get_node_with_input(workflow, input_name):
    for node_id, node in workflow.items():
        if input_name in node.get("inputs", {}):
            return node_id, node
    return None, None

def remove_missing_seed_generator_node(prompt: dict, seed: int, seed_node_id: str = "67"):
    # If node 67 exists and is a custom "Seed Generator", remove it and break links.
    node = prompt.get(seed_node_id)
    if node and node.get("class_type") == "Seed Generator":
        # Replace any connection [67, x] with the numeric seed
        for nid, n in prompt.items():
            inputs = n.get("inputs", {})
            for k, v in list(inputs.items()):
                if isinstance(v, list) and len(v) >= 1 and str(v[0]) == seed_node_id:
                    inputs[k] = int(seed)

        # Delete the problematic node
        del prompt[seed_node_id]

    return prompt

# --- Main Handler ---

def handler(job):
    job_input = job.get("input", {})
    prompt_text = job_input.get("prompt")
    negative_prompt_text = job_input.get("negative_prompt")
    width = job_input.get("width")
    height = job_input.get("height")
    steps = job_input.get("steps")
    cfg = job_input.get("cfg")
    seed = job_input.get("seed")
    lora_url = job_input.get("lora_url")
    lora_name = job_input.get("lora_name")
    lora_strength_model = job_input.get("lora_strength_model")
    lora_strength_clip = job_input.get("lora_strength_clip")
    timeout_sec = job_input.get("timeout_sec", 600)

    if not prompt_text:
        return {"error": "Missing prompt"}

    print(f"--- Job Started: {job.get('id')} ---")
    print(f"Input: Prompt='{prompt_text[:50]}...', Steps={steps}, Size={width}x{height}")

    # 1. Load Workflow
    try:
        with open(WORKFLOW_PATH, 'r') as f:
            workflow = json.load(f)
    except Exception as e:
        return {"error": f"Failed to load workflow template: {str(e)}"}

    # 2. Patch Workflow (Dynamic)
    
    # Prompt (Positive)
    # Strategy: Find CLIPTextEncode nodes. Usually Positive has 'text' input.
    # We might differentiate Positive/Negative by connection, but simple heuristic:
    # Often there are exactly two CLIPTextEncode. If we can't distiguish, we might need specific logic or just set the first one found if only 1 text provided.
    # Better generic strategy for Comfy standard prompt:
    # Usually: Node 6 (CLIPTextEncode) -> KSampler 'positive', Node 7 -> 'negative'.
    # Without IDs, we might look for 'text' input.
    
    text_nodes = get_nodes_by_class(workflow, "CLIPTextEncode")
    # Heuristic: If we find 2, assume typical ksampler setup? 
    # Or just search for one that has input 'text' with value 'CLIP_POSITIVE' placeholder?
    # User instructions say: "locate nodes by class_type of text encode (if there are several, patch the one that has input text)".
    # This is ambiguous if both have input text. 
    # Let's try to find if inputs indicate role, otherwise assume based on values if they are placeholders, or just standard 6/7 logic if keys match standard, but requirement says "No dependency on IDs".
    
    # Robust Patching:
    # Let's verify 'text' key inside 'inputs'.
    patched_positive = False
    patched_negative = False
    
    # We will try to rely on current values if they look like placeholders, OR just iterate.
    # A common convention is that the Positive prompt is the one loaded with the Checkpoint's CLIP first, 
    # but strictly speaking, without traversing the graph to the KSampler, it's hard to distinguish Positive/Negative just by class.
    # However, we can look for specific input strings if the template uses them (e.g. "positive_prompt").
    # If not, we will assume the first one found is positive (often created first) and second is negative? Risk.
    # Let's look at the actual KSampler inputs if possible.
    
    # Let's try to traverse from KSampler
    ksampler_id, ksampler = get_node_by_class(workflow, "KSampler")
    if not ksampler_id:
        ksampler_id, ksampler = get_node_by_class(workflow, "SamplerCustom") # Fallback for advanced
    
    if ksampler:
        # Trace positive/negative inputs
        pos_link = ksampler.get("inputs", {}).get("positive", [])
        neg_link = ksampler.get("inputs", {}).get("negative", [])
        
        if pos_link and len(pos_link) > 0:
            # Format usually [node_id, slot_index]
            pos_node_id = pos_link[0]
            if workflow.get(pos_node_id):
                # The node connected to 'positive' of KSampler is often a conditioning node, or the TextEncode itself.
                # If it's pure TextEncode, patch it.
                if "TextEncode" in workflow[pos_node_id].get("class_type", ""):
                     workflow[pos_node_id]["inputs"]["text"] = prompt_text
                     patched_positive = True
        
        if neg_link and len(neg_link) > 0:
            neg_node_id = neg_link[0]
            if workflow.get(neg_node_id):
                 if "TextEncode" in workflow[neg_node_id].get("class_type", "") and negative_prompt_text:
                     workflow[neg_node_id]["inputs"]["text"] = negative_prompt_text
                     patched_negative = True

    # Fallback if graph tracing failed (e.g. complex conditioning combination)
    if not patched_positive:
        # Just find any TextEncode and patch the first one?
        # User Instruction: "localize nodes by class_type... patch the one that has input text"
        # We'll just patch all of them? No, that would overwrite negative.
        # Let's assume the template has some sanity.
        # We will assume the FIRST retrieved is positive.
        if len(text_nodes) > 0:
            workflow[text_nodes[0][0]]["inputs"]["text"] = prompt_text
            if len(text_nodes) > 1 and negative_prompt_text:
                workflow[text_nodes[1][0]]["inputs"]["text"] = negative_prompt_text

    # Dimensions (Width/Height)
    # "locate the node that has inputs width and height"
    empty_latent_id, empty_latent = get_node_with_input(workflow, "width")
    if empty_latent and "height" in empty_latent["inputs"]:
        if width: empty_latent["inputs"]["width"] = width
        if height: empty_latent["inputs"]["height"] = height

    # Seed
    # "locate node with input seed or sampler with seed direct"
    seed_node_id, seed_node = get_node_with_input(workflow, "seed")
    # Also check "noise_seed" (standard in some nodes)
    if not seed_node_id:
         seed_node_id, seed_node = get_node_with_input(workflow, "noise_seed")
    
    final_seed = seed if seed else random.randint(1, 999999999999999)
    
    if seed_node:
        if "seed" in seed_node["inputs"]:
            seed_node["inputs"]["seed"] = final_seed
        elif "noise_seed" in seed_node["inputs"]:
            seed_node["inputs"]["noise_seed"] = final_seed
    
    # --- FIX: Removed missing Seed Generator and inject seed directly ---
    seed = int(final_seed)
    
    # Set seed directly on known samplers (IDs 35, 36) if they exist
    for sampler_id in ("35", "36"):
        if sampler_id in workflow and "inputs" in workflow[sampler_id]:
            workflow[sampler_id]["inputs"]["seed"] = seed
            
    # Remove missing custom node "Seed Generator" and replace its links with numeric seed
    workflow = remove_missing_seed_generator_node(workflow, seed, "67")
    # ------------------------------------------------------------------
    
    # Steps/CFG
    # "locate sampler nodes"
    samplers = get_nodes_by_class(workflow, "KSampler")
    if not samplers: samplers = get_nodes_by_class(workflow, "Sampler") # Catch-all
    
    for s_id, s_node in samplers:
        if steps and "steps" in s_node["inputs"]:
            s_node["inputs"]["steps"] = steps
        if cfg and "cfg" in s_node["inputs"]:
            s_node["inputs"]["cfg"] = cfg

    # LoRA
    if lora_url:
        # 1. Download
        if not lora_name:
             # Extract from URL
             parsed = urlparse(lora_url)
             lora_name = os.path.basename(parsed.path) or f"lora_{uuid.uuid4().hex[:8]}.safetensors"
        
        lora_disk_path = f"/comfyui/models/loras/{lora_name}"
        if not os.path.exists(lora_disk_path):
            try:
                download_file(lora_url, lora_disk_path)
            except Exception as e:
                print(f"Warning: Failed to download LoRA: {e}")
        
        # 2. Patch Node
        lora_node_id = LORA_NODE_ID
        lora_node = None
        
        if lora_node_id and workflow.get(lora_node_id):
            lora_node = workflow[lora_node_id]
        else:
            # Find by class
            l_id, l_node = get_node_by_class(workflow, "LoraLoader")
            if l_node:
                lora_node = l_node
        
        if lora_node:
            lora_node["inputs"]["lora_name"] = lora_name
            if lora_strength_model is not None:
                lora_node["inputs"]["strength_model"] = lora_strength_model
            if lora_strength_clip is not None:
                lora_node["inputs"]["strength_clip"] = lora_strength_clip
            print(f"Patched LoRA node with {lora_name}")

    # 3. Execution ComfyUI
    client_id = str(uuid.uuid4())
    ws = WebSocket()
    try:
        ws.connect(f"{COMFY_WS_URL}?clientId={client_id}")
    except Exception as e:
        return {"error": f"Failed to connect to ComfyUI WS: {str(e)}"}

    # Queue Prompt
    payload = {"prompt": workflow, "client_id": client_id}
    try:
        req = requests.post(f"{COMFY_API_URL}/prompt", json=payload)
        req.raise_for_status()
        prompt_response = req.json()
        prompt_id = prompt_response.get("prompt_id")
        print(f"Queued Prompt ID: {prompt_id}")
    except Exception as e:
        return {"error": f"Failed to queue prompt: {str(e)}"}

    # Wait for completion
    files_to_upload = []
    start_time = time.time()
    
    while True:
        if time.time() - start_time > timeout_sec:
             return {"error": "Timeout waiting for generation"}
        
        try:
            out = ws.recv()
            if isinstance(out, str):
                msg = json.loads(out)
                msg_type = msg.get("type")
                
                if msg_type == "executing":
                    data = msg.get("data", {})
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        print("Execution complete.")
                        break # Done!
                elif msg_type == "execution_error":
                     data = msg.get("data", {})
                     if data.get("prompt_id") == prompt_id:
                         return {"error": f"ComfyUI Error: {data.get('exception_message')}"}
        except Exception as e:
            return {"error": f"WebSocket Error: {str(e)}"}

    # 4. Get Outputs
    # Prefer /history
    try:
        hist_req = requests.get(f"{COMFY_API_URL}/history/{prompt_id}")
        hist_req.raise_for_status()
        history = hist_req.json().get(prompt_id, {})
        outputs = history.get("outputs", {})
        
        # Extract images
        for node_id, node_output in outputs.items():
            imgs = node_output.get("images", [])
            for img in imgs:
                files_to_upload.append(img)
                
    except Exception as e:
        print(f"Warning: Failed to fetch history: {e}")

    # Fallback to filesystem if no images found in history API
    if not files_to_upload:
        # Look for latest file in output dir
        out_dir = "/comfyui/output"
        if os.path.exists(out_dir):
            files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(('.png', '.jpg', '.webp'))]
            if files:
                latest_file = max(files, key=os.path.getctime)
                # Mock an object to fit standard processing
                files_to_upload.append({
                    "filename": os.path.basename(latest_file),
                    "subfolder": "",
                    "type": "output",
                    "_local_path": latest_file # Marker for direct read
                })

    if not files_to_upload:
        return {"error": "No output images found."}

    # 5. Download & Upload
    results = []
    for file_info in files_to_upload:
        filename = file_info.get("filename")
        
        # Get bytes
        img_data = None
        if file_info.get("_local_path"):
            with open(file_info["_local_path"], "rb") as f:
                img_data = f.read()
        else:
            # Download from Comfy View API
            params = {
                "filename": filename,
                "subfolder": file_info.get("subfolder", ""),
                "type": file_info.get("type", "output")
            }
            res = requests.get(f"{COMFY_API_URL}/view", params=params)
            if res.status_code == 200:
                img_data = res.content
        
        if img_data:
            # Upload to Supabase
            new_filename = f"{SUPABASE_PATH_PREFIX}/{prompt_id}_{filename}"
            print(f"Uploading to {SUPABASE_BUCKET}/{new_filename}...")
            
            try:
                # Supabase Storage Upload
                res = supabase.storage.from_(SUPABASE_BUCKET).upload(
                    path=new_filename,
                    file=img_data,
                    file_options={"content-type": "image/png"} # Assume png for now, or detect
                )
                
                # Construct Response
                # Check availability (if public)
                # We can't easily check public status via API without trying to get public URL
                public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(new_filename)
                
                # Check if it looks accessible (optional, or just return it)
                results.append({
                    "url": public_url,
                    "bucket": SUPABASE_BUCKET,
                    "path": new_filename
                })
                
            except Exception as e:
                print(f"Upload failed: {e}")
                results.append({"error": f"Upload failed for {filename}", "details": str(e)})

    return {"images": results, "prompt_id": prompt_id}

runpod.serverless.start({"handler": handler})
