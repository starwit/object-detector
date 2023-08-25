FROM python:3.10-slim as build

# Populate poetry artifact cache with torch libs
ADD "https://download.pytorch.org/whl/cu118/torchvision-0.15.2%2Bcu118-cp310-cp310-linux_x86_64.whl" /root/.cache/pypoetry/artifacts/8a/5a/f2/17840e3e7de0bb7049fb034c6b3404bc24827c49ebd5e893da356d42ba/
ADD "https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl" /root/.cache/pypoetry/artifacts/a8/65/bc/e9ab7708c15c5bf2697ba7610b1de40aa2ad0fafbc4780299910205883/

# Download all variants of ultralytics yolov8
ADD "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt" /code/
ADD "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt" /code/
ADD "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt" /code/
ADD "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt" /code/
ADD "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt" /code/

RUN apt update && apt install --no-install-recommends -y \
    curl \
    git \
    build-essential

ARG POETRY_VERSION
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# Copy only files that are necessary to install dependencies
COPY poetry.lock poetry.toml pyproject.toml /code/

WORKDIR /code
RUN --mount=type=secret,id=GIT_CREDENTIALS,target=/root/.git-credentials \
    git config --global credential.helper store && \
    poetry install
    
# Copy the rest of the project
COPY . /code/


### Main artifact / deliverable image

FROM python:3.10-slim
RUN apt update && apt install --no-install-recommends -y \
    libglib2.0-0 \
    libgl1
COPY --from=build /code /code
WORKDIR /code
ENV PATH="/code/.venv/bin:$PATH"
CMD [ "python", "run_stage.py" ]