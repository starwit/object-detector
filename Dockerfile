FROM starwitorg/base-python-image:0.0.15 AS build

# Download all variants of ultralytics yolov8
ADD "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt" /code/
ADD "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt" /code/
ADD "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt" /code/
ADD "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt" /code/
ADD "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt" /code/

# This is needed for `tensorrt-*` installation (see https://github.com/NVIDIA/TensorRT/issues/3050)
ENV NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=True

# Copy only files that are necessary to install dependencies
COPY poetry.lock poetry.toml pyproject.toml /code/
WORKDIR /code
RUN poetry install
    
# Copy the rest of the project
COPY . /code/


### Main artifact / deliverable image

FROM python:3.12-slim
RUN apt update && apt install --no-install-recommends -y \
    libglib2.0-0 \
    libgl1 \
    libturbojpeg0

# Create a non-root user and group
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

COPY --from=build --chown=appuser:appgroup /code /code
WORKDIR /code

# Switch to non-root user
USER appuser

ENV PATH="/code/.venv/bin:$PATH"
CMD [ "python", "main.py" ]
