#!/bin/bash

docker build -t starwitorg/sae-object-detector:$(poetry version --short) .