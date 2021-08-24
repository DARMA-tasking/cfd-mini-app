
ARG arch=amd64
FROM ${arch}/ubuntu:21.04 as base

ARG proxy=""
ARG compiler=clang-9

ENV https_proxy=${proxy} \
    http_proxy=${proxy}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y -q && \
    apt-get install -y -q --no-install-recommends \
    ca-certificates \
    curl \
    less \
    git \
    wget \
    ${compiler} \
    zlib1g \
    zlib1g-dev \
    ninja-build \
    unzip \
    valgrind \
    make-guile \
    libomp5 \
    libomp-dev \
    libvtk9-dev \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    googletest \
    libgtest-dev \
    libgmock-dev \
    ccache && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s \
    "$(which $(echo ${compiler}  | cut -d- -f1)++-$(echo ${compiler}  | cut -d- -f2))" \
    /usr/bin/clang++

ENV CC=${compiler} \
    CXX=clang++

COPY ./ci/deps/cmake.sh cmake.sh
RUN ./cmake.sh 3.18.4

ENV PATH=/cmake/bin/:$PATH
ENV LESSCHARSET=utf-8

COPY ./ci/deps/kokkos.sh kokkos.sh
RUN ./kokkos.sh 3.1.01 /pkgs 0
ENV KOKKOS_ROOT=/pkgs/kokkos/install/lib/cmake/Kokkos/

COPY ./ci/deps/kokkos-kernels.sh kokkos-kernels.sh
RUN ./kokkos-kernels.sh 3.2.00 /pkgs
ENV KOKKOS_KERNELS_ROOT=/pkgs/kokkos-kernels/install/lib/cmake/KokkosKernels/

FROM base as build
COPY . /cfd-mini-app

RUN /cfd-mini-app/ci/build_cpp.sh /cfd-mini-app/cppfd /build

FROM build as test
RUN /cfd-mini-app/ci/test_cpp.sh /cfd-mini-app/cppfd /build
