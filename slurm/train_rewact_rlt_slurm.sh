#!/bin/bash
#SBATCH --job-name=rewact_rlt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -euo pipefail

module purge
module load brics/apptainer-multi-node

home_dir="/home/u6cr/pravsels.u6cr"
scratch_dir="/scratch/u6cr/pravsels.u6cr"
repo_dir="${home_dir}/gym-hil"
rewact_dir="${repo_dir}/external/rewact"
data_dir="${scratch_dir}/gym-hil"
container="${data_dir}/container/gym_hil_rewact_arm64.sif"
config_file="${rewact_dir}/configs/train_rlt.yaml"
output_dir="${data_dir}/outputs/rewact_rlt_build_block_tower"
python_pkg_root="${data_dir}/python_packages"
hf_cache="${scratch_dir}/huggingface_cache"
wandb_dir="${data_dir}/wandb"
wandb_cache_dir="${scratch_dir}/.cache/wandb"
wandb_config_dir="${scratch_dir}/.config/wandb"
xdg_cache_home="${scratch_dir}/.cache"
xdg_config_home="${scratch_dir}/.config"
torch_home="${scratch_dir}/.cache/torch"
repo_token_file="${rewact_dir}/.hf_token"
home_token_file="${home_dir}/.hf_token"
extra_train_args="${EXTRA_TRAIN_ARGS:-}"
resume_args=""

rewact_rev="$(git -C "${rewact_dir}" rev-parse HEAD)"
robocandy_rev="$(git -C "${repo_dir}/external/robocandywrapper" rev-parse HEAD)"
python_pkg_dir="${python_pkg_root}/${rewact_rev}-${robocandy_rev}"
python_pkg_stamp="${python_pkg_dir}/.install-complete"

if [ -n "${LOAD_CKPT_PATH:-}" ]; then
    resume_args="--resume=true --checkpoint_path=${LOAD_CKPT_PATH}"
fi

mkdir -p \
    "${data_dir}/container" \
    "${output_dir}" \
    "${python_pkg_root}" \
    "${hf_cache}" \
    "${wandb_dir}" \
    "${wandb_cache_dir}" \
    "${wandb_config_dir}" \
    "${xdg_cache_home}" \
    "${xdg_config_home}" \
    "${torch_home}"

if [ ! -d "${repo_dir}" ]; then
    echo "ERROR: repo_dir not found: ${repo_dir}"
    exit 1
fi

if [ ! -d "${rewact_dir}" ]; then
    echo "ERROR: rewact_dir not found: ${rewact_dir}"
    echo "Did you clone submodules with git submodule update --init --recursive?"
    exit 1
fi

if [ ! -f "${config_file}" ]; then
    echo "ERROR: config file not found: ${config_file}"
    exit 1
fi

if [ ! -f "${container}" ]; then
    echo "ERROR: container image not found: ${container}"
    exit 1
fi

if [ -f "${repo_token_file}" ]; then
    token_file="${repo_token_file}"
elif [ -f "${home_token_file}" ]; then
    token_file="${home_token_file}"
else
    echo "ERROR: no Hugging Face token file found."
    echo "Expected one of:"
    echo "  ${repo_token_file}"
    echo "  ${home_token_file}"
    exit 1
fi

start_time="$(date -Is --utc)"

echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started (UTC): ${start_time}"
echo "Repo: ${repo_dir}"
echo "RewACT dir: ${rewact_dir}"
echo "Config: ${config_file}"
echo "Output dir: ${output_dir}"
echo "Container: ${container}"
echo "Resume args: ${resume_args:-<none>}"
echo "Extra train args: ${extra_train_args:-<none>}"
echo "===================================="

train_cmd="python scripts/train.py --config=configs/train_rlt.yaml --policy.device=cuda --output_dir=${output_dir} ${resume_args} ${extra_train_args}"
install_cmd="mkdir -p ${python_pkg_dir} && rm -rf ${python_pkg_dir:?}/* && python -m pip install --no-deps --upgrade --target ${python_pkg_dir} ${rewact_dir}/rewact_tools ${rewact_dir}/lerobot_policy_rewact ${rewact_dir}/lerobot_policy_actvantage ${repo_dir}/external/robocandywrapper"

export_cmds="export PYTHONUNBUFFERED=1"
export_cmds="${export_cmds} && export WANDB_MODE=offline"
export_cmds="${export_cmds} && export WANDB_DIR=${wandb_dir}"
export_cmds="${export_cmds} && export WANDB_CACHE_DIR=${wandb_cache_dir}"
export_cmds="${export_cmds} && export WANDB_CONFIG_DIR=${wandb_config_dir}"
export_cmds="${export_cmds} && export XDG_CACHE_HOME=${xdg_cache_home}"
export_cmds="${export_cmds} && export XDG_CONFIG_HOME=${xdg_config_home}"
export_cmds="${export_cmds} && export TORCH_HOME=${torch_home}"
export_cmds="${export_cmds} && export HF_HOME=/root/.cache/huggingface"
export_cmds="${export_cmds} && export HF_TOKEN=\$(cat ${token_file})"
export_cmds="${export_cmds} && export HUGGING_FACE_HUB_TOKEN=\${HF_TOKEN}"
export_cmds="${export_cmds} && export PYTHONPATH=${python_pkg_dir}:\${PYTHONPATH:-}"

echo "Running training command..."
echo "Command: ${train_cmd}"
echo

set +e
srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${rewact_dir}" \
    --bind "${repo_dir}:${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --bind "${hf_cache}:/root/.cache/huggingface" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "${container}" \
    bash -lc "${export_cmds} && if [ ! -f \"${python_pkg_stamp}\" ]; then ${install_cmd} && date -Is > \"${python_pkg_stamp}\"; fi && ${train_cmd}"
exit_code=$?
set -e

end_time="$(date -Is --utc)"

echo
echo "===================================="
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
echo "Exit Code: ${exit_code}"
echo "===================================="

if [ ${exit_code} -ne 0 ]; then
    echo "ERROR: training failed with exit code ${exit_code}"
    echo "Check slurm-${SLURM_JOB_ID}.err for details."
    exit ${exit_code}
fi
