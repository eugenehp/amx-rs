fn main() {
    cc::Build::new()
        .file("c/amx.c")
        .opt_level(2)
        .compile("amx");
}
