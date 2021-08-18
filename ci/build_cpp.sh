#!/usr/bin/env bash

set -ex

source_dir=${1}
build_dir=${2}

echo -e "===\n=== ccache statistics before build\n==="
ccache -s

mkdir -p "${build_dir}"
pushd "${build_dir}"

export CFD=${source_dir}
export CFD_BUILD=${build_dir}/cfd-mini-app
mkdir -p "$CFD_BUILD"
cd "$CFD_BUILD"
rm -Rf ./*

cmake -G "${CMAKE_GENERATOR:-Ninja}" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
      -DCMAKE_CXX_COMPILER="${CXX:-c++}" \
      -DCMAKE_C_COMPILER="${CC:-cc}" \
      -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS:-}" \
      -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}" \
      -DCMAKE_INSTALL_PREFIX="$CFD_BUILD/install" \
      -DKokkos_DIR="${KOKKOS_ROOT}" \
      -DKokkosKernels_DIR="${KOKKOS_KERNELS_ROOT}" \
      "$CFD"

time cmake --build . --target "${target}"

echo -e "===\n=== ccache statistics after build\n==="
ccache -s
