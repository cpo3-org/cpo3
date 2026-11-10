fn main() {
    println!("cargo:rerun-if-changed=src/linear_solvers/accelerate.c");
    #[cfg(feature = "accelerate")]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        cc::Build::new()
            .file("src/linear_solvers/accelerate.c")
            .std("c11")
            .compile("cpo3_accelerate");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    #[cfg(feature = "cholmod")]
    {
        println!("cargo:rerun-if-changed=src/linear_solvers/cholmod.c");
        // Only use the headers; emitting link metadata would make CHOLMOD
        // mandatory when loading cpo3, even if another backend is selected.
        let mut config = pkg_config::Config::new();
        config.cargo_metadata(false).range_version("5.2".."5.4");
        let library = config
            .probe("cholmod")
            .or_else(|_| config.probe("CHOLMOD"))
            .expect("the cholmod feature requires CHOLMOD 5.2.x or 5.3.x headers; set PKG_CONFIG_PATH to its pkgconfig directory");
        cc::Build::new()
            .file("src/linear_solvers/cholmod.c")
            .includes(library.include_paths)
            .std("c11")
            .compile("cpo3_cholmod");
    }
}
