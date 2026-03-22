//! Persistent AMX thread pool.
//!
//! Each worker packs A+B locally and computes in a single C call.
//! B packing is redundant across workers but cache-friendly (L1-hot).

#[cfg(all(feature = "std", target_arch = "aarch64"))]
mod inner {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Once;
    use core::cell::UnsafeCell;
    use crate::matrix::{TILE, TILE_BYTES, aligned_alloc, aligned_free};

    const MAX_WORKERS: usize = 32;
    const MAX_K_BUF: usize = 2048;
    const MAX_N_TILES: usize = 128;

    #[repr(C, align(128))]
    struct WorkerSlot {
        /// Per-worker generation: dispatcher bumps to wake this specific worker
        work_gen: AtomicU32,
        done_gen: AtomicU32,
        job: UnsafeCell<SgemmJob>,
        a_pack: *mut u8,
        z_buf: *mut u8,
    }
    unsafe impl Send for WorkerSlot {}
    unsafe impl Sync for WorkerSlot {}

    #[repr(C)]
    struct SgemmJob {
        a: usize, lda: usize,
        b_packed: usize,
        c: usize, ldc: usize,
        m: usize, k: usize, n: usize,
        tile_start: usize, tile_end: usize,
        b_ready_flag: usize, // pointer to b_ready AtomicU32
        b_ready_gen: u32,    // generation to wait for
    }

    struct AmxPool {
        slots: Vec<WorkerSlot>,
        n_workers: usize,
        b_pack: *mut u8,
        b_pack_size: usize,
    }

    static mut POOL: *const AmxPool = core::ptr::null();
    static POOL_INIT: Once = Once::new();

    fn get_pool() -> &'static AmxPool {
        unsafe {
            POOL_INIT.call_once(|| {
                let n_cpus = std::thread::available_parallelism()
                    .map(|n| n.get()).unwrap_or(1).min(MAX_WORKERS);
                let n_workers = if n_cpus > 1 { n_cpus - 1 } else { 0 };

                let a_sz = MAX_K_BUF * TILE_BYTES;
                let b_pack_size = MAX_N_TILES * MAX_K_BUF * TILE_BYTES;

                let mut slots = Vec::with_capacity(n_workers);
                for _ in 0..n_workers {
                    // z_buf sized for max n_j_tiles × 16 rows × 64 bytes
                    let z_sz = MAX_N_TILES * 16 * TILE_BYTES;
                    slots.push(WorkerSlot {
                        work_gen: AtomicU32::new(0),
                        done_gen: AtomicU32::new(0),
                        job: UnsafeCell::new(core::mem::zeroed()),
                        a_pack: aligned_alloc(a_sz, 128),
                        z_buf: aligned_alloc(z_sz, 128),
                    });
                }

                let pool = Box::leak(Box::new(AmxPool {
                    slots, n_workers,
                    b_pack: aligned_alloc(b_pack_size, 128),
                    b_pack_size,
                }));
                POOL = pool as *const AmxPool;

                for i in 0..n_workers {
                    let pool_ptr = POOL as usize;
                    std::thread::Builder::new()
                        .name(format!("amx-{i}"))
                        .spawn(move || worker_main(pool_ptr, i))
                        .expect("spawn amx worker");
                }
            });
            &*POOL
        }
    }

    fn worker_main(pool_ptr: usize, idx: usize) {
        let pool = unsafe { &*(pool_ptr as *const AmxPool) };
        let slot = &pool.slots[idx];

        #[cfg(target_os = "macos")]
        unsafe {
            extern "C" { fn pthread_set_qos_class_self_np(q: u32, p: i32) -> i32; }
            let _ = pthread_set_qos_class_self_np(0x21, 0);
        }

        // Keep AMX enabled permanently — no per-dispatch amx_set/clr overhead
        unsafe { amx_sys::amx_set(); }

        let mut my_gen = 0u32;
        loop {
            // Spin on THIS worker's per-worker generation
            loop {
                let g = slot.work_gen.load(Ordering::Relaxed);
                if g > my_gen { my_gen = g; break; }
                core::hint::spin_loop();
            }
            std::sync::atomic::fence(Ordering::Acquire);

            let job = unsafe { &*slot.job.get() };
            if job.tile_start < job.tile_end {
                unsafe {
                    // AMX stays set permanently — no amx_set/clr per dispatch
                    amx_sys::amx_sgemm_worker(
                        job.a as *const f32, job.lda as i32,
                        job.b_packed as *const f32, job.ldc as i32,
                        job.c as *mut f32, job.ldc as i32,
                        job.m as i32, job.k as i32, job.n as i32,
                        job.tile_start as i32, job.tile_end as i32,
                        slot.a_pack, slot.z_buf,
                        job.b_ready_gen as i32,
                    );
                }
            }

            slot.done_gen.store(my_gen, Ordering::Release);
        }
    }

    pub(crate) unsafe fn pool_sgemm(
        a: *const f32, lda: usize,
        b: *const f32, ldb: usize,
        c: *mut f32, ldc: usize,
        m: usize, k: usize, n: usize,
    ) {
        let n_i_tiles = (m + TILE - 1) / TILE;

        let pool = get_pool();
        let nw = pool.n_workers;

        // Check if B can be loaded directly (requires 64-byte alignment)
        let b_aligned = (b as usize) % 64 == 0 && (ldb * 4) % 64 == 0;
        // n must be multiple of 16 for direct B (no zero-padding needed)
        let direct_b = b_aligned && n % 16 == 0 && n <= 256;

        if nw == 0 || n_i_tiles < 2 {
            if !direct_b {
                amx_sys::amx_pack_b(b, ldb as i32, k as i32, n as i32, pool.b_pack);
            }
            let n_j_tiles = (n + 15) / 16;
            let z_sz = n_j_tiles * 16 * TILE_BYTES;
            let a_buf = aligned_alloc(MAX_K_BUF * TILE_BYTES, 128);
            let z_buf = aligned_alloc(z_sz, 128);
            amx_sys::amx_set();
            amx_sys::amx_sgemm_worker(
                a, lda as i32,
                if direct_b { b } else { pool.b_pack as *const f32 },
                if direct_b { ldb as i32 } else { 0 },
                c, ldc as i32,
                m as i32, k as i32, n as i32,
                0, n_i_tiles as i32,
                a_buf, z_buf,
                if direct_b { 1 } else { 0 },
            );
            amx_sys::amx_clr();
            aligned_free(a_buf, MAX_K_BUF * TILE_BYTES, 128);
            aligned_free(z_buf, z_sz, 128);
            return;
        }

        // Pack B on main thread (only if not using direct B)
        if !direct_b {
            amx_sys::amx_pack_b(b, ldb as i32, k as i32, n as i32, pool.b_pack);
        }

        let active = nw.min(n_i_tiles);
        let rows_per = (n_i_tiles + active - 1) / active;

        // Write jobs ONLY to active workers
        for i in 0..active {
            let slot = &pool.slots[i];
            let job = &mut *slot.job.get();
            let start = i * rows_per;
            let end = ((i + 1) * rows_per).min(n_i_tiles);
            *job = SgemmJob {
                a: a as usize, lda,
                b_packed: if direct_b { b as usize } else { pool.b_pack as usize },
                c: c as usize, ldc,
                m, k, n,
                tile_start: start, tile_end: end,
                b_ready_flag: 0,
                b_ready_gen: if direct_b { 1 } else { 0 },
            };
        }

        // Bump ONLY active workers' per-worker generations
        for i in 0..active {
            pool.slots[i].work_gen.fetch_add(1, Ordering::Release);
        }

        // Wait ONLY for active workers
        for i in 0..active {
            let slot = &pool.slots[i];
            let target = slot.work_gen.load(Ordering::Relaxed);
            loop {
                if slot.done_gen.load(Ordering::Acquire) >= target { break; }
                core::hint::spin_loop();
            }
        }
    }
}

#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub(crate) use inner::pool_sgemm;

#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub(crate) unsafe fn pool_dispatch_tiles(
    a_packed: *const u8, b_packed: *const u8,
    c_out: *mut f32, m: usize, k: usize, n: usize,
) {
    let z = crate::matrix::aligned_alloc(16 * crate::matrix::TILE_BYTES, 128);
    amx_sys::amx_set();
    amx_sys::amx_f32_tile_loop(a_packed, b_packed, c_out, z,
        m as i32, k as i32, n as i32, 0,
        (((m+15)/16)*((n+15)/16)) as i32);
    amx_sys::amx_clr();
    crate::matrix::aligned_free(z, 16 * crate::matrix::TILE_BYTES, 128);
}
