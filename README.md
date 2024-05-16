# Object Detector

This component is part of the Starwit Awareness Engine (SAE). See umbrella repo here: https://github.com/starwit/starwit-awareness-engine

## How to set up
- Make sure you have Poetry installed (otherwise see here: https://python-poetry.org/docs/#installing-with-the-official-installer)
- Set environment variable `NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=True` (otherwise `tensorrt-*` installation will fail)
- Run `poetry install`

# Nvidia TensorRT
A model optimized with Nvidia TensorRT can be much faster (200-300% increase in throughput), but is a lot more difficult to handle
compare to the pytorch model, which is very portable.\
Only set `use_tensorrt` in the model config to `True` if you know what you are doing!\
Other detector settings strongly depend on the parameters that were used for the TensorRT optimizization, e.g. inference size and batch size.\
Also, the installed Nvidia driver and CUDA component versions need to be tightly managed.