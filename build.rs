fn main() {
    println!("cargo:rerun-if-changed=src/linear_solvers/accelerate.c");
    #[cfg(feature = "accelerate")]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        cc::Build::new()
            .file("src/linear_solvers/accelerate.c")
            .flag("-std=c11")
            .compile("cpo3_accelerate");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    #[cfg(feature = "cholmod")]
    {
        println!("cargo:rerun-if-changed=src/linear_solvers/cholmod.c");
        let library = pkg_config::Config::new()
            .atleast_version("5.2")
            .probe("cholmod")
            .or_else(|_| pkg_config::Config::new().atleast_version("5.2").probe("CHOLMOD"))
            .expect("the cholmod feature requires CHOLMOD >= 5.2; set PKG_CONFIG_PATH to its pkgconfig directory");
        cc::Build::new()
            .file("src/linear_solvers/cholmod.c")
            .includes(library.include_paths)
            .flag("-std=c11")
            .compile("cpo3_cholmod");
    }
}
