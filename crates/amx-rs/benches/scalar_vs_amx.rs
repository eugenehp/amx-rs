//! Comprehensive benchmark: Scalar Rust vs AMX vs AMX Parallel vs Apple Accelerate
//!
//! Run:
//!   cargo bench -p amx-rs --bench scalar_vs_amx -- --nocapture
//!
//! Env vars:
//!   BENCH_ITERS=20          iterations per measurement (default 20)
//!   BENCH_MAX_N=512         largest square dimension (default 512)
//!   BENCH_CSV=path.csv      also write machine-readable CSV output

use amx::Matrix;
use amx::Vector;
use std::hint::black_box;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Apple Accelerate / BLAS FFI
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
type CBLAS_ORDER = i32;
#[allow(non_camel_case_types)]
type CBLAS_TRANSPOSE = i32;
const CBLAS_ROW_MAJOR: CBLAS_ORDER = 101;
const CBLAS_NO_TRANS: CBLAS_TRANSPOSE = 111;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: CBLAS_ORDER, transa: CBLAS_TRANSPOSE, transb: CBLAS_TRANSPOSE,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
    fn cblas_sdot(n: i32, x: *const f32, incx: i32, y: *const f32, incy: i32) -> f32;
}

fn blas_sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), n as i32,
            0.0, c.as_mut_ptr(), n as i32,
        );
    }
}

fn blas_sdot(a: &[f32], b: &[f32]) -> f32 {
    unsafe { cblas_sdot(a.len() as i32, a.as_ptr(), 1, b.as_ptr(), 1) }
}

// ---------------------------------------------------------------------------
// Timing / helpers
// ---------------------------------------------------------------------------

fn bench_fn<F: FnMut()>(mut f: F, iters: u32) -> f64 {
    for _ in 0..3 { f(); }
    let start = Instant::now();
    for _ in 0..iters { f(); }
    start.elapsed().as_secs_f64() / iters as f64
}

fn adaptive_iters(base: u32, n: usize) -> u32 {
    if n <= 8 { base * 50 }
    else if n <= 32 { base * 10 }
    else if n <= 128 { base * 3 }
    else { base }
}

fn make_f32_data(len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect()
}
fn make_matrix(m: usize, n: usize) -> Matrix<f32> {
    Matrix::from_data(make_f32_data(m * n), m, n).unwrap()
}
fn make_vector(n: usize) -> Vector<f32> {
    Vector::from_data(make_f32_data(n))
}

fn fmt_us(us: f64) -> String {
    if us < 1.0 { format!("{:>8.3}", us) }
    else if us < 1000.0 { format!("{:>8.1}", us) }
    else { format!("{:>8.0}", us) }
}
fn fmt_gf(gf: f64) -> String {
    if gf < 0.01 { format!("{:>6.3}", gf) }
    else if gf < 10.0 { format!("{:>6.2}", gf) }
    else if gf < 1000.0 { format!("{:>6.1}", gf) }
    else { format!("{:>6.0}", gf) }
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

struct CsvRow {
    chip: String, op: String, shape: String,
    m: usize, k: usize, n: usize, flops: f64,
    scalar_us: f64, amx_us: f64, amx_par_us: f64, blas_us: f64,
    scalar_gf: f64, amx_gf: f64, amx_par_gf: f64, blas_gf: f64,
}

impl CsvRow {
    fn header() -> &'static str {
        "chip,op,shape,m,k,n,flops,scalar_us,amx_us,amx_par_us,blas_us,scalar_gflops,amx_gflops,amx_par_gflops,blas_gflops"
    }
    fn to_csv(&self) -> String {
        format!("{},{},{},{},{},{},{:.0},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            self.chip, self.op, self.shape, self.m, self.k, self.n, self.flops,
            self.scalar_us, self.amx_us, self.amx_par_us, self.blas_us,
            self.scalar_gf, self.amx_gf, self.amx_par_gf, self.blas_gf)
    }
}

// ---------------------------------------------------------------------------
// Matmul measurement
// ---------------------------------------------------------------------------

// Returns: (scalar_us, smart_us, gebp_us, gebp_par_us, blas_us,
//           gf_scalar, gf_smart, gf_gebp, gf_gebp_par, gf_blas)
fn measure_matmul(
    m: usize, k: usize, n: usize,
    base_iters: u32, n_threads: usize, has_amx: bool,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let a = make_matrix(m, k);
    let b = make_matrix(k, n);
    let a_raw = make_f32_data(m * k);
    let b_raw = make_f32_data(k * n);
    let mut c_raw = vec![0.0f32; m * n];
    let size_metric = ((m * k * n) as f64).cbrt() as usize;
    let iters = adaptive_iters(base_iters, size_metric.max(m.max(n)));
    let flops = 2.0 * m as f64 * k as f64 * n as f64;

    let t_scalar = bench_fn(|| { black_box(a.matmul_scalar(&b).unwrap()); }, iters);

    // Smart dispatch (now routes to GEBP)
    let t_smart = if has_amx {
        bench_fn(|| { black_box(a.matmul(&b).unwrap()); }, iters)
    } else { t_scalar };

    // GEBP single-threaded
    let t_gebp = if has_amx {
        bench_fn(|| { black_box(a.matmul_gebp(&b).unwrap()); }, iters)
    } else { t_scalar };

    // GEBP parallel
    let t_gebp_par = if has_amx && n_threads > 1 {
        bench_fn(|| { black_box(a.matmul_gebp_parallel(&b, n_threads).unwrap()); }, iters)
    } else { t_gebp };

    let t_blas = bench_fn(|| {
        c_raw.fill(0.0);
        blas_sgemm(&a_raw, &b_raw, &mut c_raw, m, k, n);
        black_box(&c_raw);
    }, iters);

    let gf_s = flops / t_scalar / 1e9;
    let gf_smart = flops / t_smart / 1e9;
    let gf_g = flops / t_gebp / 1e9;
    let gf_gp = flops / t_gebp_par / 1e9;
    let gf_b = flops / t_blas / 1e9;

    (t_scalar * 1e6, t_smart * 1e6, t_gebp * 1e6, t_gebp_par * 1e6, t_blas * 1e6,
     gf_s, gf_smart, gf_g, gf_gp, gf_b)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let base_iters: u32 = std::env::var("BENCH_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let max_n: usize = std::env::var("BENCH_MAX_N")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let csv_path = std::env::var("BENCH_CSV").ok();
    let has_amx = amx_sys::is_amx_available();
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    let chip = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "Unknown".into());

    let mut csv_rows: Vec<CsvRow> = Vec::new();

    println!();
    println!("╔════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  Scalar vs AMX vs AMX-parallel vs Accelerate — {:<50}║", chip);
    println!("║  iters={:<4} max_n={:<4} threads={:<3} amx={:<62}║",
        base_iters, max_n, n_threads, if has_amx { "yes" } else { "no" });
    println!("╚════════════════════════════════════════════════════════════════════════════════════════════════════╝");

    // ── SQUARE MATMUL ────────────────────────────────────────────────
    let sq_sizes: Vec<usize> = [1,2,4,8,15,16,17,31,32,33,63,64,65,
                                 127,128,129,255,256,257,511,512,1024]
        .iter().copied().filter(|&n| n <= max_n).collect();

    println!();
    println!("  SQUARE MATMUL f32 (N×N × N×N)");
    println!("  {:>5}  {:>8} {:>8} {:>8} {:>8} {:>8}  {:>6} {:>6} {:>6} {:>6} {:>6}",
        "N", "scalar", "smart", "GEBP", "GEBP-par", "Accel", "GFs", "GFsm", "GFg", "GFgp", "GFbl");
    println!("  {}", "─".repeat(110));

    for &sz in &sq_sizes {
        let (su, sm, au, apu, bu, gs, gsm, ga, gap, gb) =
            measure_matmul(sz, sz, sz, base_iters, n_threads, has_amx);

        println!("  {:>5}  {} {} {} {} {}  {} {} {} {} {}",
            sz, fmt_us(su), fmt_us(sm), fmt_us(au), fmt_us(apu), fmt_us(bu),
            fmt_gf(gs), fmt_gf(gsm), fmt_gf(ga), fmt_gf(gap), fmt_gf(gb));

        csv_rows.push(CsvRow {
            chip: chip.clone(), op: "matmul".into(),
            shape: format!("{sz}x{sz}x{sz}"), m: sz, k: sz, n: sz,
            flops: 2.0 * (sz as f64).powi(3),
            scalar_us: su, amx_us: au, amx_par_us: apu, blas_us: bu,
            scalar_gf: gs, amx_gf: ga, amx_par_gf: gap, blas_gf: gb,
        });
    }

    // ── RECTANGULAR MATMUL ───────────────────────────────────────────
    let rect: Vec<(usize,usize,usize)> = vec![
        (256,1,256),(256,4,256),(256,16,256),(512,1,512),(1024,1,1),
        (1,256,256),(4,256,256),(16,256,256),
        (64,64,1),(128,128,1),(256,256,1),
        (1,1,256),(1,1,1024),
        (17,33,19),(37,53,41),(100,200,150),
        (1,768,768),(32,768,768),(128,768,3072),
        (1,4096,4096),(32,4096,4096),
    ].into_iter().filter(|&(m,k,n)| m<=max_n*2 && k<=max_n*8 && n<=max_n*8).collect();

    println!();
    println!("  RECTANGULAR MATMUL f32 (M×K × K×N)");
    println!("  {:>16}  {:>8} {:>8} {:>8} {:>8} {:>8}  {:>6} {:>6} {:>6} {:>6} {:>6}",
        "shape", "scalar", "smart", "GEBP", "GEBP-par", "Accel", "GFs", "GFsm", "GFg", "GFgp", "GFbl");
    println!("  {}", "─".repeat(118));

    for &(m,k,n) in &rect {
        let (su, sm, au, apu, bu, gs, gsm, ga, gap, gb) =
            measure_matmul(m, k, n, base_iters, n_threads, has_amx);

        println!("  {:>16}  {} {} {} {} {}  {} {} {} {} {}",
            format!("{m}×{k}×{n}"),
            fmt_us(su), fmt_us(sm), fmt_us(au), fmt_us(apu), fmt_us(bu),
            fmt_gf(gs), fmt_gf(gsm), fmt_gf(ga), fmt_gf(gap), fmt_gf(gb));

        csv_rows.push(CsvRow {
            chip: chip.clone(), op: "matmul".into(),
            shape: format!("{m}x{k}x{n}"), m, k, n,
            flops: 2.0 * m as f64 * k as f64 * n as f64,
            scalar_us: su, amx_us: au, amx_par_us: apu, blas_us: bu,
            scalar_gf: gs, amx_gf: ga, amx_par_gf: gap, blas_gf: gb,
        });
    }

    // ── DOT PRODUCT ──────────────────────────────────────────────────
    let dot_sizes = vec![1,4,16,64,256,1024,4096,16384,65536,262144,1048576];

    println!();
    println!("  DOT PRODUCT f32 (a · b)");
    println!("  {:>8}  {:>8} {:>8} {:>8} {:>8}  {:>6} {:>6} {:>6} {:>6}",
        "length", "scalar", "NEON", "AMX", "Accel", "GFs", "GFn", "GFa", "GFbl");
    println!("  {}", "─".repeat(86));

    for &sz in &dot_sizes {
        let a = make_vector(sz);
        let b = make_vector(sz);
        let a_raw = make_f32_data(sz);
        let b_raw = make_f32_data(sz);
        let iters = adaptive_iters(base_iters * 5, sz);
        let flops = 2.0 * sz as f64;

        let t_s = bench_fn(|| { black_box(a.dot_scalar(&b).unwrap()); }, iters);
        let t_n = if has_amx {
            bench_fn(|| { black_box(a.dot_neon(&b).unwrap()); }, iters)
        } else { t_s };
        let t_a = if has_amx {
            bench_fn(|| { black_box(a.dot_amx(&b).unwrap()); }, iters)
        } else { t_s };
        let t_b = bench_fn(|| { black_box(blas_sdot(&a_raw, &b_raw)); }, iters);

        let gs = flops/t_s/1e9; let gn = flops/t_n/1e9;
        let ga = flops/t_a/1e9; let gb = flops/t_b/1e9;

        println!("  {:>8}  {} {} {} {}  {} {} {} {}",
            sz, fmt_us(t_s*1e6), fmt_us(t_n*1e6), fmt_us(t_a*1e6), fmt_us(t_b*1e6),
            fmt_gf(gs), fmt_gf(gn), fmt_gf(ga), fmt_gf(gb));

        csv_rows.push(CsvRow {
            chip: chip.clone(), op: "dot".into(),
            shape: format!("len={sz}"), m: sz, k: 1, n: 1, flops,
            scalar_us: t_s*1e6, amx_us: t_a*1e6, amx_par_us: t_n*1e6, blas_us: t_b*1e6,
            scalar_gf: gs, amx_gf: ga, amx_par_gf: gn, blas_gf: gb,
        });
    }

    // ── CSV ──────────────────────────────────────────────────────────
    if let Some(path) = csv_path {
        if let Some(parent) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut out = String::from(CsvRow::header());
        out.push('\n');
        for row in &csv_rows { out.push_str(&row.to_csv()); out.push('\n'); }
        std::fs::write(&path, &out).expect("failed to write CSV");
        println!("\n  CSV → {path}");
    }

    println!();
}
