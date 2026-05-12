fn main() {
    cc::Build::new()
        .cpp(true)
        .file("shim/shim.cc")
        .include("/usr/local/include")
        .include("/usr/local/include/libint2")
        .include("/usr/include/eigen3")
        .flag("-std=c++17")
        .flag("-O2")
        .compile("ferric_shim");

    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=static=int2");
    println!("cargo:rustc-link-lib=dylib=openblas");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed=shim/shim.h");
    println!("cargo:rerun-if-changed=shim/shim.cc");
}
