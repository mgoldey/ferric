# libint2: conda-forge 2.13.1 (prebuilt, second-derivative integrals), installed
# by scripts/install-libint.sh into /opt/libint2. The script sets the library's
# SONAME to /opt/libint2/lib/libint2.so, so the runtime stage must keep that path.

# Stage 1: Build ferric
FROM ubuntu:22.04 AS ferric-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake g++ gfortran wget ca-certificates curl \
    libeigen3-dev libopenblas-dev liblapack-dev pkg-config \
    libopenmpi-dev openmpi-bin libxc-dev libboost-dev patchelf zstd \
    python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/install-libint.sh /tmp/install-libint.sh
RUN /tmp/install-libint.sh /opt/libint2
ENV LIBINT2_PREFIX=/opt/libint2

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin
RUN pip3 install --no-cache-dir maturin numpy

# Copy source
WORKDIR /ferric
COPY . .

# Build workspace with MPI
RUN cargo build --release --workspace --features mpi

# Run tests
RUN cargo test --workspace --features mpi -- --test-threads=1

# Build Python bindings
RUN cd crates/ferric-python \
    && maturin build --release \
    && pip3 install --no-cache-dir target/wheels/ferric-*.whl

# Verify Python bindings
RUN python3 -c "import ferric; mol = ferric.Molecule.from_xyz('testdata/molecules/water.xyz'); print(f'natoms={mol.natoms()}')"

# Runtime stage (optional slim image)
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 liblapack3 libgomp1 python3 python3-pip \
    libopenmpi-dev openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ferric-builder /opt/libint2/lib/libint2.so /opt/libint2/lib/libint2.so
COPY --from=ferric-builder /ferric/target/release/ferric-cli /usr/local/bin/ferric
COPY --from=ferric-builder /usr/local/lib/python3/dist-packages /usr/local/lib/python3/dist-packages
RUN ldconfig

WORKDIR /work
ENTRYPOINT ["ferric"]
