#!/usr/bin/env sh

set -eu

python_bin="${PYTHON:-python3}"
build_dir=".build"
root_egg_info="pnmi.egg-info"
build_egg_info="$build_dir/$root_egg_info"

mkdir -p "$build_dir"

if ! "$python_bin" -c 'import setuptools' >/dev/null 2>&1; then
    echo "install_local.sh: setuptools is required in the selected Python environment" >&2
    echo "use a virtual environment where setuptools is already installed" >&2
    exit 1
fi

"$python_bin" -m pip install -e . --no-build-isolation
if [ -d "$root_egg_info" ]; then
    rm -rf "$build_egg_info"
    mv "$root_egg_info" "$build_egg_info"
fi
