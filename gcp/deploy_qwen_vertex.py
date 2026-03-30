"""Deploy Qwen3.5-35B-A3B to Vertex AI endpoint for max throughput.

Model: Qwen/Qwen3.5-35B-A3B (MoE — 35B total / 3B active per token)
BF16 weights: ~70GB total → TP=4 required (17.5GB per H100)
Context: 131072 tokens (native Qwen max)
Thinking mode: enabled via --reasoning-parser qwen3

VRAM budget per GPU (H100 80GB, gpu-memory-utilization=0.90):
  Usable: 72GB | Weights: 17.5GB | KV cache: ~54GB
  KV cache per token: ~25KB/GPU
  131072 tokens × 32 seqs × 25KB ≈ 105GB → fits across 2 GPUs; safe with TP=4

H100 quota:
  europe-west4: 48 GPUs  → a3-highgpu-4g, up to 12 replicas (TP=4)
  europe-west1: 48 GPUs  → a3-highgpu-8g, up to 6 replicas (TP=8, overkill)
  us-central1:  16 GPUs  → a3-highgpu-4g, up to 4 replicas (TP=4)

Throughput: TP=4, max-num-seqs=32 → 32 concurrent requests per replica
  (reduced from 64 to support 131072 context; 200-300 theorem proofs use ~25/replica)
  8 replicas → 256 max concurrent  (comfortable for 200-300 prompts)
  4 replicas → 128 max concurrent  (minimum for 200-300 prompts)

Usage:
  python scripts/deploy_qwen_vertex.py --region europe-west4 --tp 4 --gpus 4 --replicas 8
  python scripts/deploy_qwen_vertex.py --region us-central1  --tp 4 --gpus 4 --replicas 4
  python scripts/deploy_qwen_vertex.py --endpoint-id <ID> --undeploy-model-id <ID>  # redeploy to existing
"""

import argparse
from google.cloud import aiplatform

parser = argparse.ArgumentParser()
parser.add_argument("--region", default="europe-west4")
parser.add_argument("--endpoint-id", default=None, help="Existing endpoint ID (creates new if not set)")
parser.add_argument("--undeploy-model-id", default=None, help="Deployed model ID to undeploy first")
parser.add_argument("--replicas", type=int, default=8)
parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size (must match --gpus)")
parser.add_argument("--gpus", type=int, default=4, help="GPUs per replica (4=a3-highgpu-4g, 8=a3-highgpu-8g)")
parser.add_argument("--deploy-timeout", type=int, default=3600, help="Deploy timeout in seconds")
parser.add_argument("--no-enforce-eager", action="store_true", default=True, help="Disable --enforce-eager (enables CUDA graphs; faster but may be less stable for MoE)")
args = parser.parse_args()

aiplatform.init(project="ax-baku", location=args.region)

VLLM_ARGS = [
    "--host=0.0.0.0",
    "--port=8080",
    "--model=Qwen/Qwen3.5-35B-A3B",
    f"--tensor-parallel-size={args.tp}",
    "--max-model-len=131072",       # native Qwen max; safe with max-num-seqs=32 on H100×4
    "--max-num-seqs=32",           # reduced from 64 to fit 131072-token KV cache in ~54GB/GPU
    "--gpu-memory-utilization=0.90",
    "--enable-chunked-prefill",
    "--enable-prefix-caching",
    "--reasoning-parser=qwen3",      # enables thinking mode (<think>...</think> extraction)
    "--enable-auto-tool-choice",     # required for function calling
    "--tool-call-parser=hermes",     # Hermes format required for Qwen3 function calling
]

if not args.no_enforce_eager:
    VLLM_ARGS.append("--enforce-eager")  # disable CUDA graphs for MoE stability

# Step 1: Upload model
print(f"Uploading model in {args.region} (TP={args.tp})...")
model = aiplatform.Model.upload(
    display_name="qwen3-5-35b-a3b",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20260320_0916_RC01",
    serving_container_command=["python", "-m", "vllm.entrypoints.openai.api_server"],
    serving_container_args=VLLM_ARGS,
    serving_container_health_route="/health",
    serving_container_predict_route="/v1/chat/completions",
    serving_container_ports=[8080],
    serving_container_environment_variables={
        "DEPLOY_SOURCE": "API_HF_VERIFIED_MODEL",
        "MODEL_ID": "Qwen/Qwen3.5-35B-A3B",
    },
)
print(f"Model uploaded: {model.resource_name}")

# Step 2: Get or create endpoint
if args.endpoint_id:
    endpoint = aiplatform.Endpoint(args.endpoint_id)
    if args.undeploy_model_id:
        print(f"Undeploying old model {args.undeploy_model_id}...")
        endpoint.undeploy(deployed_model_id=args.undeploy_model_id)
        print("Old model undeployed.")
else:
    print("Creating new endpoint...")
    endpoint = aiplatform.Endpoint.create(
        display_name="qwen3-5-35b-a3b",
        dedicated_endpoint_enabled=True,
        inference_timeout=7200,
    )
    print(f"Endpoint created: {endpoint.resource_name}")

# Step 3: Deploy
print(f"Deploying with {args.replicas} replicas ({args.replicas * args.gpus} H100s total)...")
endpoint.deploy(
    model=model,
    machine_type=f"a3-highgpu-{args.gpus}g",
    accelerator_type="NVIDIA_H100_80GB",
    accelerator_count=args.gpus,
    min_replica_count=args.replicas,
    max_replica_count=args.replicas,
    deploy_request_timeout=args.deploy_timeout,
)
print(f"Deployed to endpoint: {endpoint.resource_name}")
print(f"Dedicated DNS: {endpoint.dedicated_endpoint_dns}")
print(f"Config: TP={args.tp}, {args.replicas} replicas, {args.replicas * 64} max concurrent requests")
print()
print("Next step: update configs/llms.yaml qwen_3_5.provider_config.base_url:")
dns = endpoint.dedicated_endpoint_dns
# Extract project number and region from the dedicated DNS:
# format: <endpoint_id>.<region>-<project_number>.prediction.vertexai.goog
parts = dns.split(".")
endpoint_id = parts[0]
region_project = parts[1]  # e.g. asia-southeast1-136811191949
region = "-".join(region_project.split("-")[:-1])
project_number = region_project.split("-")[-1]
vertex_base_url = f"https://{dns}/v1beta1/projects/{project_number}/locations/{region}/endpoints/{endpoint_id}"
print(f"  base_url: \"{vertex_base_url}\"")
