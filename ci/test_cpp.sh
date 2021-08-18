#!/usr/bin/env bash

set -exo pipefail

source_dir=${1}
build_dir=${2}

export CFD=${source_dir}
export CFD_BUILD=${build_dir}/cfd-mini-app
pushd "$CFD_BUILD"

# Once ctest is working do this
# ctest --output-on-failure | tee cmake-output.log

./test

popd
