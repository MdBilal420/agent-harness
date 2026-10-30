From python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# --- System CLIs ---
# Install common utilities the agent may use via bash. We do this in a
# single RUN command to keep the image layer count small and to allow apt's
# cache to be cleaned up in the same layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    jq \
    unzip \
    zip \
    less \
    tree \
    ripgrep \
    fd-find \
    sqlite3 \
    openssl \
    build-essential \
 && rm -rf /var/lib/apt/lists/*


# --- Symlink fd-find to fd ---
RUN ln -s /usr/bin/fdfind /usr/local/bin/fd


# --- Node.js ---
# Install Node.js for the agent to use.
 RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

# --- Workspace ---
# Set the workspace directory for the agent to use.
 WORKDIR /workspace
