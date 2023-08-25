#!/bin/bash

docker push docker.internal.starwit-infra.de/starwit/object-detector:$(poetry version --short)