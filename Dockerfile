# Stage 1: Build libint2 from mpqc4 tarball
FROM ubuntu:22.04 AS libint-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake g++ gfortran wget ca-certificates \
    libeigen3-dev libopenblas-dev liblapack-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN wget -q https://github.com/evaleev/libint/releases/download/v2.7.2/libint-2.7.2-mpqc4.tgz \
    && tar xzf libint-2.7.2-mpqc4.tgz \
    && cd libint-2.7.2-mpqc4 \
    && mkdir build && cd build \
    && cmake .. \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && make install

# Stage 2: Build ferric
FROM ubuntu:22.04 AS ferric-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake g++ gfortran wget ca-certificates curl \
    libeigen3-dev libopenblas-dev liblapack-dev pkg-config \
    python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Copy libint2 from stage 1
COPY --from=libint-builder /usr/local/lib/libint2* /usr/local/lib/
COPY --from=libint-builder /usr/local/include/libint2 /usr/local/include/libint2
COPY --from=libint-builder /usr/local/include/libint2.h /usr/local/include/libint2.h
COPY --from=libint-builder /usr/local/include/libint2.hpp /usr/local/include/libint2.hpp
COPY --from=libint-builder /usr/local/lib/cmake/libint2 /usr/local/lib/cmake/libint2
COPY --from=libint-builder /usr/local/lib/pkgconfig/libint2.pc /usr/local/lib/pkgconfig/libint2.pc
RUN ldconfig

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin
RUN pip3 install --no-cache-dir maturin numpy

# Copy source
WORKDIR /ferric
COPY . .

# Build workspace
RUN cargo build --release --workspace

# Run tests
RUN cargo test --workspace -- --test-threads=1

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
    && rm -rf /var/lib/apt/lists/*

COPY --from=ferric-builder /usr/local/lib/libint2* /usr/local/lib/
COPY --from=ferric-builder /ferric/target/release/ferric-cli /usr/local/bin/ferric
COPY --from=ferric-builder /usr/local/lib/python3/dist-packages /usr/local/lib/python3/dist-packages
RUN ldconfig

WORKDIR /work
ENTRYPOINT ["ferric"]
