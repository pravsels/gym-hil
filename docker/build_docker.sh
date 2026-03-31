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

PLATFORM="${1:-$(detect_platform)}"
IMAGE_NAME="${IMAGE_NAME:-gym-hil-rewact}"
IMAGE_TAG="${IMAGE_TAG:-${PLATFORM}}"
BUILD_NETWORK="${DOCKER_BUILD_NETWORK:-host}"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} for linux/${PLATFORM}"

docker buildx build \
    --platform "linux/${PLATFORM}" \
    --network "${BUILD_NETWORK}" \
    --load \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "${REPO_DIR}/docker/Dockerfile" \
    "${REPO_DIR}"
