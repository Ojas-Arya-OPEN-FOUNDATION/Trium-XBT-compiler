fn main() {
    println!("cargo:rerun-if-env-changed=BT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=BT_RUNTIME_DIR");
    if let Ok(dir) = std::env::var("BT_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", dir);
    }
    if let Ok(dir) = std::env::var("BT_RUNTIME_DIR") {
        println!("cargo:rustc-link-search=native={}", dir);
    }
}
