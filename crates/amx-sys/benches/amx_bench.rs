//! Per-core, per-instruction, and whole-chip AMX benchmarks for Apple Silicon.
//!
//! Measures:
//!   1. Single P-core: ns/op, GFLOPS, Ginstr/s
//!   2. Single E-core: same (on heterogeneous chips)
//!   3. P vs E side-by-side
//!   4. All P-cores parallel: aggregate GFLOPS, Ginstr/s, per-core ns/op
//!   5. All E-cores parallel: same
//!   6. Whole chip (P+E): aggregate GFLOPS + Ginstr/s
//!
//! Uses macOS QoS classes to steer P vs E:
//!   - QOS_CLASS_USER_INTERACTIVE (0x21) → P-core
//!   - QOS_CLASS_BACKGROUND       (0x09) → E-core

use amx_sys::*;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

// ---------------------------------------------------------------------------
// macOS QoS
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
type qos_class_t = u32;
const QOS_CLASS_USER_INTERACTIVE: qos_class_t = 0x21;
const QOS_CLASS_BACKGROUND: qos_class_t = 0x09;

extern "C" {
    fn pthread_set_qos_class_self_np(qos: qos_class_t, relative_priority: i32) -> i32;
}

fn set_qos(qos: qos_class_t) {
    unsafe { pthread_set_qos_class_self_np(qos, 0); }
}

// ---------------------------------------------------------------------------
// System detection
// ---------------------------------------------------------------------------

fn sysctl_str(key: &str) -> String {
    std::process::Command::new("sysctl")
        .args(["-n", key])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn sysctl_u32(key: &str) -> u32 {
    sysctl_str(key).parse().unwrap_or(0)
}

fn detect_chip() -> String {
    let s = sysctl_str("machdep.cpu.brand_string");
    if s.is_empty() { "Unknown Apple Silicon".into() } else { s }
}

fn detect_core_counts() -> (u32, u32) {
    if sysctl_u32("hw.nperflevels") >= 2 {
        (sysctl_u32("hw.perflevel0.logicalcpu"),
         sysctl_u32("hw.perflevel1.logicalcpu"))
    } else {
        (sysctl_u32("hw.logicalcpu"), 0)
    }
}

// ---------------------------------------------------------------------------
// Benchmark kernels
// ---------------------------------------------------------------------------

fn black_box<T>(x: T) -> T {
    unsafe { std::ptr::read_volatile(&x) }
}

#[repr(align(128))]
struct A64([u8; 64]);

fn bench_set_clr(iters: u64) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        unsafe { amx_set(); amx_clr(); }
    }
    start.elapsed().as_secs_f64() / iters as f64
}

fn bench_ldx_stx(iters: u64) -> f64 {
    let mut buf = A64([0u8; 64]);
    unsafe { amx_set(); }
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            amx_ldx(ptr_row_flags(buf.0.as_ptr(), 0, 0));
            amx_stx(ptr_row_flags(buf.0.as_mut_ptr(), 0, 0));
        }
    }
    let e = start.elapsed().as_secs_f64() / iters as f64;
    unsafe { amx_clr(); }
    black_box(&buf);
    e
}

macro_rules! make_bench {
    ($name:ident, $setup:expr, $op:ident, $operand:expr) => {
        fn $name(iters: u64) -> f64 {
            let mut x_buf = A64([0u8; 64]);
            let mut y_buf = A64([0u8; 64]);
            let mut z_buf = A64([0u8; 64]);
            #[allow(clippy::redundant_closure_call)]
            ($setup)(&mut x_buf.0, &mut y_buf.0);
            unsafe {
                amx_set();
                amx_ldx(ptr_row_flags(x_buf.0.as_ptr(), 0, 0));
                amx_ldy(ptr_row_flags(y_buf.0.as_ptr(), 0, 0));
            }
            let operand: u64 = $operand;
            let start = Instant::now();
            for _ in 0..iters {
                unsafe { $op(operand); }
            }
            let e = start.elapsed().as_secs_f64() / iters as f64;
            unsafe {
                amx_stz(ptr_row_flags(z_buf.0.as_mut_ptr(), 0, 0));
                amx_clr();
            }
            black_box(&z_buf);
            e
        }
    };
}

fn fill_f16(x: &mut [u8; 64], y: &mut [u8; 64]) {
    for i in 0..32 {
        x[i*2..i*2+2].copy_from_slice(&0x3C00u16.to_le_bytes());
        y[i*2..i*2+2].copy_from_slice(&0x3C00u16.to_le_bytes());
    }
}
fn fill_f32(x: &mut [u8; 64], y: &mut [u8; 64]) {
    for i in 0..16 {
        x[i*4..i*4+4].copy_from_slice(&1.0f32.to_le_bytes());
        y[i*4..i*4+4].copy_from_slice(&1.0f32.to_le_bytes());
    }
}
fn fill_f64(x: &mut [u8; 64], y: &mut [u8; 64]) {
    for i in 0..8 {
        x[i*8..i*8+8].copy_from_slice(&1.0f64.to_le_bytes());
        y[i*8..i*8+8].copy_from_slice(&1.0f64.to_le_bytes());
    }
}
fn fill_i16(x: &mut [u8; 64], y: &mut [u8; 64]) {
    for i in 0..32 {
        x[i*2..i*2+2].copy_from_slice(&1i16.to_le_bytes());
        y[i*2..i*2+2].copy_from_slice(&2i16.to_le_bytes());
    }
}

make_bench!(bench_fma16_vec, fill_f16, amx_fma16, 1u64 << 63);
make_bench!(bench_fma16_mat, fill_f16, amx_fma16, 0u64);
make_bench!(bench_fma32_vec, fill_f32, amx_fma32, 1u64 << 63);
make_bench!(bench_fma32_mat, fill_f32, amx_fma32, 0u64);
make_bench!(bench_fma64_vec, fill_f64, amx_fma64, 1u64 << 63);
make_bench!(bench_fma64_mat, fill_f64, amx_fma64, 0u64);
make_bench!(bench_mac16_vec, fill_i16, amx_mac16, 1u64 << 63);
make_bench!(bench_mac16_mat, fill_i16, amx_mac16, 0u64);

// ---------------------------------------------------------------------------
// Metric helpers
// ---------------------------------------------------------------------------

fn gflops_from_ns(ns_per_op: f64, ops_per_instr: u64) -> f64 {
    if ops_per_instr == 0 || ns_per_op <= 0.0 { return 0.0; }
    (ops_per_instr as f64) / ns_per_op
}

fn ginstr_from_ns(ns_per_op: f64) -> f64 {
    if ns_per_op <= 0.0 { return 0.0; }
    1.0 / ns_per_op
}

// ---------------------------------------------------------------------------
// Bench table definition
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct BenchResult {
    name: &'static str,
    ns: f64,
    ops_per_instr: u64,
}

const ALL_BENCHES: &[(&str, fn(u64) -> f64, u64)] = &[
    ("set + clr",              bench_set_clr,   0),
    ("ldx + stx",              bench_ldx_stx,   0),
    ("fma16 vector  (32 f16)", bench_fma16_vec,  2 * 32),
    ("fma16 matrix  (32×32)",  bench_fma16_mat,  2 * 32 * 32),
    ("fma32 vector  (16 f32)", bench_fma32_vec,  2 * 16),
    ("fma32 matrix  (16×16)",  bench_fma32_mat,  2 * 16 * 16),
    ("fma64 vector  (8 f64)",  bench_fma64_vec,  2 * 8),
    ("fma64 matrix  (8×8)",    bench_fma64_mat,  2 * 8 * 8),
    ("mac16 vector  (32 i16)", bench_mac16_vec,  2 * 32),
    ("mac16 matrix  (32×32)",  bench_mac16_mat,  2 * 32 * 32),
];

/// Subset used for parallel/whole-chip tests (skip set+clr and ldx+stx overhead tests).
const COMPUTE_BENCHES: &[(&str, fn(u64) -> f64, u64)] = &[
    ("fma16 vector  (32 f16)", bench_fma16_vec,  2 * 32),
    ("fma16 matrix  (32×32)",  bench_fma16_mat,  2 * 32 * 32),
    ("fma32 vector  (16 f32)", bench_fma32_vec,  2 * 16),
    ("fma32 matrix  (16×16)",  bench_fma32_mat,  2 * 16 * 16),
    ("fma64 vector  (8 f64)",  bench_fma64_vec,  2 * 8),
    ("fma64 matrix  (8×8)",    bench_fma64_mat,  2 * 8 * 8),
    ("mac16 vector  (32 i16)", bench_mac16_vec,  2 * 32),
    ("mac16 matrix  (32×32)",  bench_mac16_mat,  2 * 32 * 32),
];

const ITERS: u64 = 2_000_000;
const PAR_ITERS: u64 = 1_000_000;

// ---------------------------------------------------------------------------
// Single-core suite
// ---------------------------------------------------------------------------

fn run_single_core_suite(label: &str) -> Vec<BenchResult> {
    let mut results = Vec::new();
    for &(name, f, ops) in ALL_BENCHES {
        f(10_000);
        let ns = f(ITERS) * 1e9;
        results.push(BenchResult { name, ns, ops_per_instr: ops });
    }

    println!();
    println!("  {label}");
    println!("  {:<30} {:>8} {:>10} {:>10}",
        "instruction", "ns/op", "Ginstr/s", "GFLOPS");
    println!("  {}", "-".repeat(60));
    for r in &results {
        let gi = ginstr_from_ns(r.ns);
        if r.ops_per_instr > 0 {
            let gf = gflops_from_ns(r.ns, r.ops_per_instr);
            println!("  {:<30} {:>8.2} {:>10.2} {:>10.2}", r.name, r.ns, gi, gf);
        } else {
            println!("  {:<30} {:>8.2} {:>10.2} {:>10}", r.name, r.ns, gi, "-");
        }
    }
    results
}

// ---------------------------------------------------------------------------
// Multi-core parallel suite
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct ParResult {
    name: &'static str,
    ops_per_instr: u64,
    per_core_ns: f64,       // worst-case per-core ns
    agg_ginstr: f64,        // aggregate Ginstr/s
    agg_gflops: f64,        // aggregate GFLOPS
}

fn measure_parallel(
    n_threads: u32,
    qos: qos_class_t,
    bench_fn: fn(u64) -> f64,
    ops_per_instr: u64,
    iters: u64,
) -> (f64, f64, f64) {
    // Returns (per_core_ns, agg_ginstr, agg_gflops)
    let barrier = Arc::new(Barrier::new(n_threads as usize + 1));
    let mut handles = Vec::new();

    for _ in 0..n_threads {
        let bar = barrier.clone();
        handles.push(thread::spawn(move || {
            set_qos(qos);
            let t = Instant::now();
            while t.elapsed().as_millis() < 20 {}
            bench_fn(5_000);
            bar.wait();
            let start = Instant::now();
            bench_fn(iters);
            start.elapsed().as_secs_f64()
        }));
    }

    barrier.wait();

    // Each thread returns its own elapsed. Sum per-thread throughput for aggregate.
    let mut agg_instr_per_sec = 0.0f64;
    let mut agg_ops_per_sec = 0.0f64;
    let mut max_elapsed = 0.0f64;

    for h in handles {
        let elapsed = h.join().unwrap();
        if elapsed > max_elapsed { max_elapsed = elapsed; }
        let thread_instr_per_sec = iters as f64 / elapsed;
        agg_instr_per_sec += thread_instr_per_sec;
        agg_ops_per_sec += thread_instr_per_sec * ops_per_instr as f64;
    }

    let per_core_ns = max_elapsed / iters as f64 * 1e9;
    let agg_ginstr = agg_instr_per_sec / 1e9;
    let agg_gflops = agg_ops_per_sec / 1e9;

    (per_core_ns, agg_ginstr, agg_gflops)
}

fn run_parallel_suite(label: &str, n_threads: u32, qos: qos_class_t) -> Vec<ParResult> {
    let mut results = Vec::new();

    for &(name, f, ops) in COMPUTE_BENCHES {
        let (ns, gi, gf) = measure_parallel(n_threads, qos, f, ops, PAR_ITERS);
        results.push(ParResult {
            name,
            ops_per_instr: ops,
            per_core_ns: ns,
            agg_ginstr: gi,
            agg_gflops: gf,
        });
    }

    println!();
    println!("  {label} ({n_threads} threads)");
    println!("  {:<30} {:>10} {:>10} {:>10}",
        "instruction", "ns/op/thr", "Ginstr/s", "GFLOPS");
    println!("  {}", "-".repeat(62));
    for r in &results {
        println!("  {:<30} {:>10.2} {:>10.2} {:>10.2}",
            r.name, r.per_core_ns, r.agg_ginstr, r.agg_gflops);
    }

    results
}

// ---------------------------------------------------------------------------
// Whole chip (mixed P+E)
// ---------------------------------------------------------------------------

fn run_whole_chip(p_cores: u32, e_cores: u32) {
    let total = p_cores + e_cores;
    let iters = PAR_ITERS;

    println!();
    println!("  Whole chip ({total} cores: {p_cores}P + {e_cores}E)");
    println!("  {:<30} {:>10} {:>10}",
        "instruction", "Ginstr/s", "GFLOPS");
    println!("  {}", "-".repeat(52));

    for &(name, bench_fn, ops) in COMPUTE_BENCHES {
        let barrier = Arc::new(Barrier::new(total as usize + 1));
        let mut handles = Vec::new();

        for _ in 0..p_cores {
            let bar = barrier.clone();
            handles.push(thread::spawn(move || {
                set_qos(QOS_CLASS_USER_INTERACTIVE);
                let t = Instant::now(); while t.elapsed().as_millis() < 20 {}
                bench_fn(5_000);
                bar.wait();
                let start = Instant::now();
                bench_fn(iters);
                start.elapsed().as_secs_f64()
            }));
        }

        for _ in 0..e_cores {
            let bar = barrier.clone();
            handles.push(thread::spawn(move || {
                set_qos(QOS_CLASS_BACKGROUND);
                let t = Instant::now(); while t.elapsed().as_millis() < 40 {}
                bench_fn(5_000);
                bar.wait();
                let start = Instant::now();
                bench_fn(iters);
                start.elapsed().as_secs_f64()
            }));
        }

        barrier.wait();

        let mut agg_instr = 0.0f64;
        let mut agg_ops = 0.0f64;
        for h in handles {
            let elapsed = h.join().unwrap();
            let thr_ips = iters as f64 / elapsed;
            agg_instr += thr_ips;
            agg_ops += thr_ips * ops as f64;
        }

        println!("  {:<30} {:>10.2} {:>10.2}",
            name, agg_instr / 1e9, agg_ops / 1e9);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    if !amx_sys::is_amx_available() {
        println!("AMX not available on this platform — skipping benchmark.");
        return;
    }

    let chip = detect_chip();
    let (p_cores, e_cores) = detect_core_counts();
    let total_cores = p_cores + e_cores;

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  AMX Benchmark — {:<47}║", chip);
    if e_cores > 0 {
        println!("║  {} P-cores + {} E-cores = {} total{:<33}║",
            p_cores, e_cores, total_cores, "");
    } else {
        println!("║  {} cores (single perf level){:<36}║", p_cores, "");
    }
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // ── 1. Single P-core ──
    set_qos(QOS_CLASS_USER_INTERACTIVE);
    let t = Instant::now(); while t.elapsed().as_millis() < 10 {}
    let p_results = run_single_core_suite("Single P-core");

    // ── 2. Single E-core ──
    let e_results = if e_cores > 0 {
        set_qos(QOS_CLASS_BACKGROUND);
        let t = Instant::now(); while t.elapsed().as_millis() < 50 {}
        Some(run_single_core_suite("Single E-core"))
    } else {
        None
    };

    // ── 3. P vs E side-by-side ──
    if let Some(ref e_res) = e_results {
        println!();
        println!("  P-core vs E-core");
        println!("  {:<30} {:>8} {:>8} {:>7}  {:>8} {:>8} {:>7}",
            "", "P ns", "P Gi/s", "P GF", "E ns", "E Gi/s", "E GF");
        println!("  {}", "-".repeat(80));
        for (p, e) in p_results.iter().zip(e_res.iter()) {
            let p_gi = ginstr_from_ns(p.ns);
            let e_gi = ginstr_from_ns(e.ns);
            if p.ops_per_instr > 0 {
                let p_gf = gflops_from_ns(p.ns, p.ops_per_instr);
                let e_gf = gflops_from_ns(e.ns, e.ops_per_instr);
                println!("  {:<30} {:>8.2} {:>8.2} {:>7.1}  {:>8.2} {:>8.2} {:>7.1}",
                    p.name, p.ns, p_gi, p_gf, e.ns, e_gi, e_gf);
            } else {
                println!("  {:<30} {:>8.2} {:>8.2} {:>7}  {:>8.2} {:>8.2} {:>7}",
                    p.name, p.ns, p_gi, "-", e.ns, e_gi, "-");
            }
        }
    }

    // ── 4. All P-cores in parallel ──
    if p_cores > 1 {
        run_parallel_suite("All P-cores parallel", p_cores, QOS_CLASS_USER_INTERACTIVE);
    }

    // ── 5. All E-cores in parallel ──
    if e_cores > 1 {
        run_parallel_suite("All E-cores parallel", e_cores, QOS_CLASS_BACKGROUND);
    }

    // ── 6. Whole chip ──
    if total_cores > 1 {
        run_whole_chip(p_cores, e_cores);
    }

    println!();
}
