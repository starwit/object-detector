# Object Detector

This component is part of the Starwit Awareness Engine (SAE). See umbrella repo here: https://github.com/starwit/vision-pipeline-k8s

## How to set up
- Make sure you have Poetry installed (otherwise see here: https://python-poetry.org/docs/#installing-with-the-official-installer)
- Until Poetry supports installing torch (see: https://github.com/python-poetry/poetry/issues/6409) we have to manually download torch and torchvision
  - For Linux just run the script `./download_torch_dependencies.sh` (it downloads the wheels from here: https://download.pytorch.org/whl/torch_stable.html)
- Run `poetry install`

## How to play around
- Download any image(s) you'd like to detection objects on (e.g. https://cocodataset.org/#explore?id=292283)
- Adapt the image path in `demo.py`
- Run `demo.py`


