#!/bin/bash

# Purpose of this script is to enforce a bash environment when calling make
export PATH=/root/.local/bin/:$PATH

make build-deb

