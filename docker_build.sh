#!/bin/bash

docker build -t starwitorg/sae-object-detector:$(git rev-parse --short HEAD) .