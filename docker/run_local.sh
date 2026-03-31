#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

detect_platform() {
    case "$(uname -m)" in
        x86_64) echo "amd64" ;;
        aarch64|arm64) echo "arm64" ;;
        *)
            echo "Unsupported host architecture: $(uname -m)" >&2
            exit 1
            ;;
    esac
}

PLATFORM="${PLATFORM:-$(detect_platform)}"
IMAGE_NAME="${IMAGE_NAME:-gym-hil-rewact}"
IMAGE_TAG="${IMAGE_TAG:-${PLATFORM}}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:---gpus all}"
DOCKER_RUN_EXTRA_ARGS="${DOCKER_RUN_EXTRA_ARGS:-}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"
TTY_ARGS=()
ENV_ARGS=()
CMD=("$@")

if [ "${#CMD[@]}" -eq 0 ]; then
    CMD=(bash)
fi

if [ -z "${HF_TOKEN_FILE}" ]; then
    if [ -f "${REPO_DIR}/external/rewact/.hf_token" ]; then
        HF_TOKEN_FILE="${REPO_DIR}/external/rewact/.hf_token"
    elif [ -f "${HOME}/.hf_token" ]; then
        HF_TOKEN_FILE="${HOME}/.hf_token"
    fi
fi

if [ -n "${HF_TOKEN_FILE}" ]; then
    if [ ! -f "${HF_TOKEN_FILE}" ]; then
        echo "HF_TOKEN_FILE does not exist: ${HF_TOKEN_FILE}" >&2
        exit 1
    fi
    ENV_ARGS+=(-e "HF_TOKEN=$(<"${HF_TOKEN_FILE}")")
    ENV_ARGS+=(-e "HUGGING_FACE_HUB_TOKEN=$(<"${HF_TOKEN_FILE}")")
fi

if [ -t 0 ] && [ -t 1 ]; then
    TTY_ARGS=(-it)
fi

echo "Running ${IMAGE_NAME}:${IMAGE_TAG} in ${CONTAINER_WORKDIR}"
echo "Command: ${CMD[*]}"

docker run --rm \
    "${TTY_ARGS[@]}" \
    ${DOCKER_GPU_ARGS} \
    ${DOCKER_RUN_EXTRA_ARGS} \
    -v "${REPO_DIR}:${CONTAINER_WORKDIR}" \
    -w "${CONTAINER_WORKDIR}" \
    "${ENV_ARGS[@]}" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${CMD[@]}"
