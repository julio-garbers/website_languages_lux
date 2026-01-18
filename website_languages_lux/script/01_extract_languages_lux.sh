#!/bin/bash -l
#SBATCH --job-name=01_extract_languages_lux
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --account=p200804
#SBATCH --qos=default
#SBATCH --error="/project/home/p200812/blog/script/slurm/01_extract_languages_lux.err"
#SBATCH --output="/project/home/p200812/blog/script/slurm/01_extract_languages_lux.out"


echo "================================================================================"
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Processing all years (2013-2024)"
echo "================================================================================"


# =============================================================================
# Project Configuration
# =============================================================================

# Base directory paths
BLOG_DIR=/project/home/p200812/blog
VLLM_DIR=/project/home/p200812/01_inskill/vllm_magistral
SCRIPT_DIR=${BLOG_DIR}/script
DATA_DIR=/project/home/p201125/firm_websites/data


# =============================================================================
# Environment Setup
# =============================================================================

# Load required modules
echo "[SETUP] Loading environment modules..."
module --force purge
module load env/release/2024
module load Apptainer/1.3.4-GCCcore-13.3.0

# Prevent Lmod and Apptainer from cluttering $HOME
export LMOD_IGNORE_CACHE=1
export APPTAINER_CACHEDIR=${VLLM_DIR}/.apptainer
export APPTAINER_TMPDIR=${VLLM_DIR}/.tmp
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# Fix PMI/munge issues on some clusters
export PMIX_MCA_psec=native


# =============================================================================
# Python Library Patching
# =============================================================================

# Set up patch directory for custom library versions
PATCH_DIR=${VLLM_DIR}/py_patch_mistral
ORIG_SIF=${VLLM_DIR}/vllm-openai_latest.sif
SIF_IMAGE=${ORIG_SIF}

mkdir -p "${PATCH_DIR}"

# Install mistral_common from GitHub main branch
echo "[PATCH] Installing mistral_common from main branch..."
apptainer exec -B ${PATCH_DIR}:/ext_patch "${ORIG_SIF}" \
    python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
       -t /ext_patch \
       "git+https://github.com/mistralai/mistral-common.git@main#egg=mistral_common"

# Install polars for data processing
echo "[PATCH] Installing polars 1.21.0..."
apptainer exec -B ${PATCH_DIR}:/ext_patch "${ORIG_SIF}" \
    python3 -m pip install --no-cache-dir -t /ext_patch "polars==1.21.0"

# Configure Python path for patched libraries
export PYTHONPATH=/ext_patch:${PYTHONPATH:-}
PY_PATCH_BIND="-B ${PATCH_DIR}:/ext_patch --env PYTHONPATH=${PYTHONPATH}"


# =============================================================================
# HuggingFace Configuration
# =============================================================================

# Set up HuggingFace cache on fast storage
echo "[HF] Configuring HuggingFace cache..."
export LOCAL_HF_CACHE=${VLLM_DIR}/HF_cache
mkdir -p "${LOCAL_HF_CACHE}"
export HF_TOKEN=add_token_here  # Replace with your actual token


# =============================================================================
# Network Configuration
# =============================================================================

# Configure hostname and IP for vLLM server
echo "[NETWORK] Setting up networking..."
export HEAD_HOSTNAME="$(hostname)"
export HEAD_IPADDRESS="$(hostname --ip-address)"
export VLLM_SERVER_URL="http://${HEAD_IPADDRESS}:8000"

echo "  Head hostname: ${HEAD_HOSTNAME}"
echo "  Head IP: ${HEAD_IPADDRESS}"
echo "  vLLM server URL: ${VLLM_SERVER_URL}"


# =============================================================================
# Container Configuration
# =============================================================================

# Configure Apptainer bind mounts and environment variables
export APPTAINER_ARGS="--nvccli ${PY_PATCH_BIND} \
  -B ${LOCAL_HF_CACHE}:${VLLM_DIR}/.cache/huggingface \
  -B ${SCRIPT_DIR}:/workspace \
  -B ${DATA_DIR}:${DATA_DIR} \
  -B ${BLOG_DIR}:${BLOG_DIR} \
  --env HF_HOME=${VLLM_DIR}/.cache/huggingface \
  --env HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} \
  --env VLLM_SERVER_URL=${VLLM_SERVER_URL}"


# =============================================================================
# Model and Parallelism Configuration
# =============================================================================

# Model settings
export HF_MODEL="mistralai/Magistral-Small-2506"
export TENSOR_PARALLEL_SIZE=4
export PIPELINE_PARALLEL_SIZE=${SLURM_NNODES}

echo "[MODEL] Configuration:"
echo "  Model: ${HF_MODEL}"
echo "  Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "  Pipeline parallel size: ${PIPELINE_PARALLEL_SIZE}"


# =============================================================================
# Ray Distributed Framework Setup
# =============================================================================

# Generate random port for Ray head node communication
export RANDOM_PORT=$(python3 - <<'PY'
import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()
PY
)
export RAY_CMD_HEAD="ray start --block --head --port=${RANDOM_PORT}"


# =============================================================================
# Launch Ray head
# =============================================================================

srun -J head -N 1 \
     --ntasks-per-node=1 -c ${SLURM_CPUS_PER_TASK} \
     apptainer exec ${APPTAINER_ARGS} ${SIF_IMAGE} ${RAY_CMD_HEAD} &

# Wait for Ray to initialize
sleep 10


# =============================================================================
# vLLM Server Launch
# =============================================================================

apptainer exec ${APPTAINER_ARGS} ${SIF_IMAGE} vllm serve ${HF_MODEL} \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
        --pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
        --gpu-memory-utilization 0.95 \
        --disable-log-requests \
        --max-model-len 16384 \
        --max-num-batched-tokens 32768 \
        --max-num-seqs 128 \
        --tokenizer-mode mistral \
        --config-format mistral \
        --load-format mistral \
        --tool-call-parser mistral \
        --enable-auto-tool-choice \
        --port 8000 &

# Wait for vLLM server to be ready
echo "[VLLM] Waiting for server to be ready at ${VLLM_SERVER_URL}..."
until curl -sSf "${VLLM_SERVER_URL}/v1/models" >/dev/null 2>&1; do
    echo "  Server not ready yet, waiting..."
    sleep 5
done
sleep 40   # Additional warm-up time
echo "[VLLM] Server is ready!"


# =============================================================================
# Inference Execution
# =============================================================================

echo "================================================================================"
echo "[INFERENCE] Starting language extraction for all years..."
echo "================================================================================"

apptainer exec ${APPTAINER_ARGS} ${SIF_IMAGE} \
     python3 /workspace/01_extract_languages_lux.py \
          --model ${HF_MODEL} \
          --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
          --pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
          --host http://${HEAD_IPADDRESS}:8000

# Capture exit status
INFERENCE_EXIT_CODE=$?

echo "================================================================================"
echo "Job completed at: $(date)"
echo "Exit code: ${INFERENCE_EXIT_CODE}"
echo "================================================================================"

exit ${INFERENCE_EXIT_CODE}