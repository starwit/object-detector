# Build & Development of object detector package
This section contains all information about how to build & run object detector for development purposes

## Docker

You can use [docker_build.sh](docker_build.sh) and [docker_publish.sh](docker_publish.sh) to build and publish Docker images. 

Once build you can run Docker image locally like so:
```bash
docker run -it --rm -v ./settings.yaml:/code/ starwitorg/starwitorg/sae-object-detector:3.0.0
```
Please note, that you should provide a settings.yaml that configures application to your needs. See [template](settings.template.yaml) for how to do that.

## APT package

This software can be released as a Debian/APT package. This section explains, how this works. Please note, that everything has been tested with Ubuntu Linux only.

Following packages need to installed on your computer:
* build-essential
* python3-all
* python3-setuptools 
* python3-pip
* dh-python

Package can be build using:
```bash
make build-deb
```

Package can be found in folder _target_. You can test install package using Docker like so:
```bash
docker run -it --rm -v ./:/app  ubuntu:22.04 bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update && apt install -y /app/target/objectdetector_0.1.0_all.deb
```

```bash
make clean
```

### Config files for packaging

Files in the `debian/` folder:
- `changelog` - Records changes made to the package in each version
- `compat` - Specifies the debhelper compatibility level
- `control` - Contains package metadata (dependencies, description, etc.)
- `install` - Lists files to be installed and their destination paths
- `objectdetector.service` - Systemd service file for the application
- `postinst` - Post-installation script that runs after package installation
- `preinst` - Pre-installation script that runs before package installation
- `prerm` - Pre-removal script that runs before package removal
- `rules` - Main package build script (makefile)
- `source/options` - Options for the source package format like ommiting directories
