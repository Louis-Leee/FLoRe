# Onboard rig image for the G1's Jetson Orin NX (L4T R35.3.1 = Ubuntu 20.04 host).
#
# Runs the F/T bridge (2_ft_bridge.sh) and the holosoma low level (3_run_policy_vive_force.sh)
# ON THE ROBOT. Ubuntu 24.04 userspace because bota-driver 1.1.9 ships only manylinux_2_34
# aarch64 wheels and the host's glibc is 2.31. The environments themselves live in the
# bind-mounted HOME (setup_envs.sh), not in the image.
#
# The robot has no internet. Load the base image from the workstation, then build ON the Jetson
# through the workstation's proxy (onboard/g1_internet.sh, reverse-forwarded to the robot's :3128):
#   workstation:  docker pull --platform linux/arm64 ubuntu:24.04
#                 docker save --platform linux/arm64 ubuntu:24.04 | ssh g1 docker load
#   robot:        docker build --network host --build-arg http_proxy=http://127.0.0.1:3128 \
#                     --build-arg https_proxy=http://127.0.0.1:3128 -t flore-rig:latest .
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.13.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential pkg-config git curl ca-certificates sudo \
        iproute2 can-utils kmod usbutils udev procps psmisc less nano tmux rsync \
        libusb-1.0-0 libgl1 libglib2.0-0 libegl1 libgles2 libosmesa6 libglu1-mesa \
        linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-aarch64-unknown-linux-gnu.tar.gz" \
        | tar -xz --strip-components=1 -C /usr/local/bin \
              uv-aarch64-unknown-linux-gnu/uv uv-aarch64-unknown-linux-gnu/uvx

# rig = uid 1000 = the Jetson's `unitree`, so files in the mounted HOME stay owned by unitree on
# the host. dialout/video/plugdev carry the host's gids (20/44/46) for the ttyACM / video / usb nodes.
# Passwordless sudo, for bringing CAN links up (`sudo ip link set <can> up`; --privileged gives the
# container NET_ADMIN).
RUN userdel -r ubuntu 2>/dev/null || true \
    && useradd -m -u 1000 -s /bin/bash -G sudo,dialout,video,plugdev rig \
    && echo 'rig ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/rig \
    && chmod 440 /etc/sudoers.d/rig

USER rig
ENV HOME=/home/rig LANG=C.UTF-8 UV_LINK_MODE=copy
WORKDIR /home/rig
