fn main() {
    cc::Build::new()
        .file("c/amx.c")
        .opt_level(3)
        .flag_if_supported("-march=native")
        .flag_if_supported("-mtune=native")
        .flag_if_supported("-funroll-loops")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-unused-variable")
        .compile("amx");
}
