//! Persistent AMX thread pool with near-zero dispatch overhead.
//!
//! Workers keep `amx_set()` active and spin-wait on atomic flags.
//! This eliminates thread spawn + AMX init cost (~20-100µs) that
//! kills parallelism for small-to-medium matrices.

#[cfg(all(feature = "std", target_arch = "aarch64"))]
mod inner {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Once;
    use core::cell::UnsafeCell;
    use crate::matrix::{TILE, TILE_BYTES, aligned_alloc, aligned_free};

    const MAX_WORKERS: usize = 32;
    // Dispatch generation: workers compare their last-seen generation to the
    // global generation to know when new work is available. This avoids the
    // IDLE/WORK_READY/WORK_DONE state machine race conditions.

    #[repr(C, align(128))] // cache-line aligned to avoid false sharing
    struct WorkerSlot {
        /// Current generation seen by this worker.  
        /// When gen < pool.generation, there's work to do.
        done_gen: AtomicU32,
        job: UnsafeCell<TileJob>,
        z_buf: *mut u8,
    }

    unsafe impl Send for WorkerSlot {}
    unsafe impl Sync for WorkerSlot {}

    #[repr(C)]
    struct TileJob {
        a_packed: usize,
        b_packed: usize,
        c_out: usize,
        m: usize,
        k: usize,
        n: usize,
        n_j_tiles: usize,
        tile_start: usize,
        tile_end: usize,
    }

    struct AmxPool {
        slots: Vec<WorkerSlot>,
        n_workers: usize,
        /// Incremented each dispatch. Workers spin until their done_gen < this.
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

                let mut slots = Vec::with_capacity(n_workers);
                for _ in 0..n_workers {
                    slots.push(WorkerSlot {
                        done_gen: AtomicU32::new(0),
                        job: UnsafeCell::new(core::mem::zeroed()),
                        z_buf: aligned_alloc(TILE * TILE_BYTES, 128),
                    });
                }

                let pool = Box::leak(Box::new(AmxPool {
                    slots,
                    n_workers,
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

        // Pin to P-core
        #[cfg(target_os = "macos")]
        unsafe {
            extern "C" { fn pthread_set_qos_class_self_np(q: u32, p: i32) -> i32; }
            let _ = pthread_set_qos_class_self_np(0x21, 0);
        }

        let mut my_gen = 0u32;
        loop {
            // Spin until new generation
            loop {
                let g = pool.generation.load(Ordering::Acquire);
                if g > my_gen { my_gen = g; break; }
                core::hint::spin_loop();
            }

            // Fence: ensure we see the job data written before generation bump
            std::sync::atomic::fence(Ordering::SeqCst);

            // Execute job if tile range is non-empty
            let job = unsafe { &*slot.job.get() };
            if job.tile_start < job.tile_end {
                unsafe {
                    amx_sys::amx_set();
                    execute_tiles(job, slot.z_buf);
                    amx_sys::amx_clr();
                }
            }

            // Signal done
            slot.done_gen.store(my_gen, Ordering::Release);
        }
    }

    #[inline(never)]
    unsafe fn execute_tiles(job: &TileJob, z_buf: *mut u8) {
        let a_packed = job.a_packed as *const u8;
        let b_packed = job.b_packed as *const u8;
        let c_out = job.c_out as *mut f32;
        let m = job.m;
        let k = job.k;
        let n = job.n;
        let n_j_tiles = job.n_j_tiles;
        let kc = 512usize;

        for idx in job.tile_start..job.tile_end {
            let it = idx / n_j_tiles;
            let jt = idx % n_j_tiles;
            let i_blk = it * TILE;
            let j_blk = jt * TILE;
            let tile_m = TILE.min(m - i_blk);
            let tile_n = TILE.min(n - j_blk);

            let ap = a_packed.add(it * k * TILE_BYTES);
            let bp = b_packed.add(jt * k * TILE_BYTES);

            if k > kc {
                let mut first = true;
                let mut ks = 0;
                while ks < k {
                    let akc = kc.min(k - ks);
                    if first {
                        amx_sys::amx_f32_tile_kernel_4y(ap.add(ks * TILE_BYTES), bp.add(ks * TILE_BYTES), z_buf, akc as i32, tile_m as i32);
                        first = false;
                    } else {
                        amx_sys::amx_f32_tile_kernel_4y_accum(ap.add(ks * TILE_BYTES), bp.add(ks * TILE_BYTES), z_buf, akc as i32, tile_m as i32);
                    }
                    ks += kc;
                }
            } else {
                amx_sys::amx_f32_tile_kernel_4y(ap, bp, z_buf, k as i32, tile_m as i32);
            }

            for ii in 0..tile_m {
                let src = z_buf.add(ii * TILE_BYTES) as *const f32;
                let dst = c_out.add((i_blk + ii) * n + j_blk);
                core::ptr::copy_nonoverlapping(src, dst, tile_n);
            }
        }
    }

    pub(crate) unsafe fn pool_dispatch_tiles(
        a_packed: *const u8,
        b_packed: *const u8,
        c_out: *mut f32,
        m: usize, k: usize, n: usize,
    ) {
        let n_i_tiles = (m + TILE - 1) / TILE;
        let n_j_tiles = (n + TILE - 1) / TILE;
        let total_tiles = n_i_tiles * n_j_tiles;

        let pool = get_pool();
        let nw = pool.n_workers;

        if nw == 0 || total_tiles < 2 {
            let z = aligned_alloc(TILE * TILE_BYTES, 128);
            amx_sys::amx_set();
            let job = TileJob {
                a_packed: a_packed as usize,
                b_packed: b_packed as usize,
                c_out: c_out as usize,
                m, k, n, n_j_tiles,
                tile_start: 0, tile_end: total_tiles,
            };
            execute_tiles(&job, z);
            amx_sys::amx_clr();
            aligned_free(z, TILE * TILE_BYTES, 128);
            return;
        }

        // Distribute ALL tiles across workers (caller just waits)
        let tiles_per = (total_tiles + nw - 1) / nw;

        // Write jobs to worker slots
        for i in 0..nw {
            let start = i * tiles_per;
            let end = ((i + 1) * tiles_per).min(total_tiles);
            let slot = &pool.slots[i];
            let job = &mut *slot.job.get();
            *job = TileJob {
                a_packed: a_packed as usize,
                b_packed: b_packed as usize,
                c_out: c_out as usize,
                m, k, n, n_j_tiles,
                tile_start: start,
                tile_end: end,
            };
        }

        // Full fence + bump generation to wake workers
        std::sync::atomic::fence(Ordering::SeqCst);
        let gen = pool.generation.fetch_add(1, Ordering::SeqCst) + 1;

        // Wait for all workers to finish this generation
        for i in 0..nw {
            let slot = &pool.slots[i];
            loop {
                if slot.done_gen.load(Ordering::Acquire) >= gen {
                    break;
                }
                core::hint::spin_loop();
            }
        }
    }
}

#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub(crate) use inner::pool_dispatch_tiles;
