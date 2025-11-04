FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

# Install make
RUN apt-get update && apt-get install -y make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pipx and ensure it's available in the PATH
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install pipx \
    && python3 -m pipx ensurepath \
    && pip install pytest-cov

# Install Java
RUN apt-get update && apt-get install -y openjdk-11-jdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
ENV OLLAMA_HOST="http://ollama:11434"

# Install Poetry
RUN pipx install poetry==1.4.2