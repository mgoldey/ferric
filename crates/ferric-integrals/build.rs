fn main() {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/matt".to_string());
    let local_prefix = format!("{home}/.local");

    cc::Build::new()
        .cpp(true)
        .file("shim/shim.cc")
        .include(format!("{local_prefix}/include"))
        .include(format!("{local_prefix}/include/libint2"))
        .include("/usr/local/include")
        .include("/usr/local/include/libint2")
        .include("/usr/include/eigen3")
        .flag("-std=c++17")
        .flag("-O2")
        .flag("-Wno-deprecated-declarations")
        .flag("-Wno-unused-parameter")
        .compile("ferric_shim");

    println!("cargo:rustc-link-search=native={local_prefix}/lib");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=static=int2");
    println!("cargo:rustc-link-lib=dylib=openblas");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed=shim/shim.h");
    println!("cargo:rerun-if-changed=shim/shim.cc");
}
