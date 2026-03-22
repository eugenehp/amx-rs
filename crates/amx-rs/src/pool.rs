//! Persistent AMX thread pool with near-zero dispatch overhead.
//!
//! Workers spin-wait on atomic generation counters. Each worker does
//! its own A/B packing + AMX compute in a single C call — no heap
//! allocations on the hot path.

#[cfg(all(feature = "std", target_arch = "aarch64"))]
mod inner {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Once;
    use core::cell::UnsafeCell;
    use crate::matrix::{TILE, TILE_BYTES, aligned_alloc, aligned_free};

    const MAX_WORKERS: usize = 32;
    /// Max K for pre-allocated packing buffers (covers most use cases)
    const MAX_K_BUF: usize = 2048;
    /// Max N tiles for pre-allocated B packing buffer
    const MAX_N_TILES: usize = 128; // covers N up to 2048

    #[repr(C, align(128))]
    struct WorkerSlot {
        done_gen: AtomicU32,
        job: UnsafeCell<SgemmJob>,
        // Pre-allocated per-worker buffers (allocated once at pool init)
        a_pack: *mut u8,  // MAX_K_BUF * TILE_BYTES
        b_pack: *mut u8,  // MAX_N_TILES * MAX_K_BUF * TILE_BYTES
        z_buf: *mut u8,   // 16 * TILE_BYTES
    }
    unsafe impl Send for WorkerSlot {}
    unsafe impl Sync for WorkerSlot {}

    #[repr(C)]
    struct SgemmJob {
        a: usize, lda: usize,
        b: usize, ldb: usize,
        c: usize, ldc: usize,
        m: usize, k: usize, n: usize,
        tile_start: usize, tile_end: usize,
    }

    struct AmxPool {
        slots: Vec<WorkerSlot>,
        n_workers: usize,
        generation: AtomicU32,
    }

    static mut POOL: *const AmxPool = core::ptr::null();
    static POOL_INIT: Once = Once::new();

    fn get_pool() -> &'static AmxPool {
        unsafe {
            POOL_INIT.call_once(|| {
                let n = std::thread::available_parallelism()
                    .map(|n| n.get()).unwrap_or(1).min(MAX_WORKERS);
                let n_workers = if n > 1 { n - 1 } else { 0 };

                let a_buf_sz = MAX_K_BUF * TILE_BYTES;
                let b_buf_sz = MAX_N_TILES * MAX_K_BUF * TILE_BYTES;

                let mut slots = Vec::with_capacity(n_workers);
                for _ in 0..n_workers {
                    slots.push(WorkerSlot {
                        done_gen: AtomicU32::new(0),
                        job: UnsafeCell::new(core::mem::zeroed()),
                        a_pack: aligned_alloc(a_buf_sz, 128),
                        b_pack: aligned_alloc(b_buf_sz, 128),
                        z_buf: aligned_alloc(16 * TILE_BYTES, 128),
                    });
                }

                let pool = Box::leak(Box::new(AmxPool {
                    slots, n_workers,
                    generation: AtomicU32::new(0),
                }));
                POOL = pool as *const AmxPool;

                for i in 0..n_workers {
                    let pool_ptr = POOL as usize;
                    let idx = i;
                    std::thread::Builder::new()
                        .name(format!("amx-{i}"))
                        .spawn(move || worker_main(pool_ptr, idx))
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

        let mut my_gen = 0u32;
        loop {
            // Tight spin (Relaxed for speed, Acquire fence after)
            loop {
                let g = pool.generation.load(Ordering::Relaxed);
                if g > my_gen { my_gen = g; break; }
                core::hint::spin_loop();
            }
            std::sync::atomic::fence(Ordering::Acquire);

            let job = unsafe { &*slot.job.get() };
            if job.tile_start < job.tile_end {
                unsafe {
                    amx_sys::amx_set();
                    amx_sys::amx_sgemm_pack_and_compute(
                        job.a as *const f32, job.lda as i32,
                        job.b as *const f32, job.ldb as i32,
                        job.c as *mut f32, job.ldc as i32,
                        job.m as i32, job.k as i32, job.n as i32,
                        job.tile_start as i32, job.tile_end as i32,
                        slot.a_pack, slot.b_pack, slot.z_buf,
                    );
                    amx_sys::amx_clr();
                }
            }

            slot.done_gen.store(my_gen, Ordering::Release);
        }
    }

    /// Dispatch sgemm across the persistent pool.
    /// Does NOT pre-pack — each worker packs its own tiles.
    pub(crate) unsafe fn pool_sgemm(
        a: *const f32, lda: usize,
        b: *const f32, ldb: usize,
        c: *mut f32, ldc: usize,
        m: usize, k: usize, n: usize,
    ) {
        let n_i_tiles = (m + TILE - 1) / TILE;
        let n_j_tiles = (n + TILE - 1) / TILE;
        let total_tiles = n_i_tiles * n_j_tiles;

        let pool = get_pool();
        let nw = pool.n_workers;

        if nw == 0 || total_tiles < 4 {
            // Single-threaded fallback
            let a_sz = MAX_K_BUF * TILE_BYTES;
            let b_sz = MAX_N_TILES * MAX_K_BUF * TILE_BYTES;
            let a_buf = aligned_alloc(a_sz, 128);
            let b_buf = aligned_alloc(b_sz, 128);
            let z_buf = aligned_alloc(16 * TILE_BYTES, 128);
            amx_sys::amx_set();
            amx_sys::amx_sgemm_pack_and_compute(
                a, lda as i32, b, ldb as i32, c, ldc as i32,
                m as i32, k as i32, n as i32,
                0, total_tiles as i32,
                a_buf, b_buf, z_buf,
            );
            amx_sys::amx_clr();
            aligned_free(a_buf, a_sz, 128);
            aligned_free(b_buf, b_sz, 128);
            aligned_free(z_buf, 16 * TILE_BYTES, 128);
            return;
        }

        // Distribute tiles across workers
        let tiles_per = (total_tiles + nw - 1) / nw;

        for i in 0..nw {
            let start = i * tiles_per;
            let end = ((i + 1) * tiles_per).min(total_tiles);
            let slot = &pool.slots[i];
            let job = &mut *slot.job.get();
            *job = SgemmJob {
                a: a as usize, lda,
                b: b as usize, ldb,
                c: c as usize, ldc,
                m, k, n,
                tile_start: start, tile_end: end,
            };
        }

        // Wake workers
        let gen = pool.generation.fetch_add(1, Ordering::Release) + 1;

        // Wait for completion
        for i in 0..nw {
            let slot = &pool.slots[i];
            loop {
                if slot.done_gen.load(Ordering::Acquire) >= gen { break; }
                core::hint::spin_loop();
            }
        }
    }
}

#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub(crate) use inner::pool_sgemm;

// Keep old API for compatibility
#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub(crate) unsafe fn pool_dispatch_tiles(
    a_packed: *const u8, b_packed: *const u8,
    c_out: *mut f32, m: usize, k: usize, n: usize,
) {
    let z_buf = crate::matrix::aligned_alloc(16 * crate::matrix::TILE_BYTES, 128);
    amx_sys::amx_set();
    amx_sys::amx_f32_tile_loop(
        a_packed, b_packed, c_out, z_buf,
        m as i32, k as i32, n as i32,
        0, (((m+15)/16) * ((n+15)/16)) as i32,
    );
    amx_sys::amx_clr();
    crate::matrix::aligned_free(z_buf, 16 * crate::matrix::TILE_BYTES, 128);
}
