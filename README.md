# Object Detector

This component is part of the Starwit Awareness Engine (SAE). See umbrella repo here: https://github.com/starwit/starwit-awareness-engine

## How to set up
- Make sure you have Poetry installed (otherwise see here: https://python-poetry.org/docs/#installing-with-the-official-installer)
- Set environment variable `NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=True` (otherwise `tensorrt-*` installation will fail)
- Run `poetry install`

## How to Build

See [dev readme](doc/DEV_README.md) for build instructions.

## Model Development
In order to detect objects a trained model is necessary. See [model development](doc/Model_Development.md) documentation for more details.
