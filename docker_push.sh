#!/bin/bash

docker push docker.internal.starwit-infra.de/sae/object-detector:$(poetry version --short)