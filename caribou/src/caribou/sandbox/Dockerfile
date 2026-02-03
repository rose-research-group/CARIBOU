# Use official NVIDIA CUDA image based on Ubuntu 22.04
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Set DEBIAN_FRONTEND to noninteractive to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# --- Install System Dependencies ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tini \
    tzdata \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libsodium-dev \
    libzmq3-dev \
    gcc \
    g++ \
    sudo \
    curl \
    wget \
    git \
    vim \
    nano \
    unzip \
    zip && \
    # Configure timezone
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    # Clean up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Install Micromamba and Create RAPIDS Environment ---
COPY ./environment.yml /tmp/environment.yml
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local bin/micromamba && \
    micromamba create -f /tmp/environment.yml -y && \
    micromamba clean -a -y

# --- Create Non-Root User & Group ---
ARG NB_USER="sandboxuser"
ARG NB_UID=1001
ARG NB_GID=1001
ENV USER=${NB_USER}
ENV HOME=/home/${NB_USER}

RUN groupadd -g ${NB_GID} ${NB_USER} && \
    useradd -m -s /bin/bash -u ${NB_UID} -g ${NB_GID} ${NB_USER} && \
    adduser ${NB_USER} sudo && \
    echo "${NB_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# --- Application Setup ---
# Copy app files and set permissions
COPY --chown=${NB_USER}:${NB_GID} ./kernel_api.py ${HOME}/kernel_api.py
COPY --chown=${NB_USER}:${NB_GID} ./start_kernel.py ${HOME}/start_kernel.py
COPY --chown=${NB_USER}:${NB_GID} ./start.sh ${HOME}/start.sh

# Create user-specific directories and make scripts executable
RUN mkdir -p ${HOME}/.local/share/jupyter \
             ${HOME}/.ipython/profile_default/startup \
             ${HOME}/.ipython/profile_default/static && \
    chmod +x ${HOME}/start_kernel.py ${HOME}/start.sh && \
    chown -R ${NB_USER}:${NB_GID} ${HOME}

# --- Runtime Configuration ---
USER ${NB_USER}
WORKDIR ${HOME}

EXPOSE 8000
ENV IPY_BASE_PORT=4000
# Activate conda environment
ENV PATH="/usr/local/envs/rapids/bin:${PATH}"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/home/sandboxuser/start.sh"]
