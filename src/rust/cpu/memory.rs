// -------------------------------------------------------------------------
// WASIP2 FORK NOTE:
//   This file diverges from upstream copy/v86 on the wasip2-rom-windows
//   branch.  The divergence adds optional ROM-window support so boot
//   providers (OVMF, netboot BIOS) can map firmware above the flat RAM
//   backing buffer.  Search for `FIRMWARE_WINDOW` / `firmware_window` for
//   the hooks; the RAM fast path is deliberately untouched so interpreter
//   performance on linux/seabios modes is unaffected.
// -------------------------------------------------------------------------

mod ext {
    extern "C" {
        pub fn mmap_read8(addr: u32) -> i32;
        pub fn mmap_read32(addr: u32) -> i32;

        pub fn mmap_write8(addr: u32, value: i32);
        pub fn mmap_write16(addr: u32, value: i32);
        pub fn mmap_write32(addr: u32, value: i32);
        pub fn mmap_write64(addr: u32, v0: i32, v1: i32);
        pub fn mmap_write128(addr: u32, v0: i32, v1: i32, v2: i32, v3: i32);
    }
}

use crate::cpu::apic;
use crate::cpu::cpu::{
    handle_irqs, reg128, APIC_MEM_ADDRESS, APIC_MEM_SIZE, IOAPIC_MEM_ADDRESS, IOAPIC_MEM_SIZE,
};
use crate::cpu::global_pointers::memory_size;
use crate::cpu::ioapic;
use crate::cpu::vga;
use crate::jit;
use crate::page::Page;

use std::alloc;
use std::ptr;

#[allow(non_upper_case_globals)]
pub static mut mem8: *mut u8 = ptr::null_mut();

// -------------------------------------------------------------------------
// Firmware window — single optional ROM region mapped above `memory_size`.
//
// The design goal is that ROM execution (OVMF, netboot BIOS, …) works
// without growing the flat `mem8` backing buffer to cover the full 4 GiB
// address space.  A firmware window is installed by the control plane
// before CPU start; reads that fall above `memory_size` land here via
// the existing MMIO dispatch chain.  Writes are currently dropped — v1
// models firmware as true ROM.  NVRAM support is future work: add a
// `writable: bool` flag and mirror writes into a separate variable
// store.
// -------------------------------------------------------------------------

pub struct FirmwareWindow {
    pub base: u32,
    pub data: Box<[u8]>,
    pub writable: bool,
}

impl FirmwareWindow {
    #[inline]
    pub fn contains(&self, addr: u32) -> bool {
        let lo = self.base as u64;
        let hi = lo + self.data.len() as u64;
        let a = addr as u64;
        a >= lo && a < hi
    }
}

#[allow(non_upper_case_globals)]
pub static mut firmware_window: Option<FirmwareWindow> = None;

/// Install a firmware window.  The previous window (if any) is dropped.
/// Call BEFORE starting the CPU — installing while the CPU is running
/// is undefined because the MMIO dispatch chain reads `firmware_window`
/// without synchronisation.
pub fn install_firmware_window(base: u32, data: Vec<u8>, writable: bool) {
    unsafe {
        firmware_window = Some(FirmwareWindow {
            base,
            data: data.into_boxed_slice(),
            writable,
        });
    }
}

pub fn clear_firmware_window() {
    unsafe {
        firmware_window = None;
    }
}

#[inline]
#[allow(static_mut_refs)]
pub fn in_firmware_window(addr: u32) -> bool {
    unsafe {
        match firmware_window.as_ref() {
            Some(fw) => fw.contains(addr),
            None => false,
        }
    }
}

#[inline]
#[allow(static_mut_refs)]
unsafe fn firmware_read8(addr: u32) -> i32 {
    let fw = firmware_window.as_ref().unwrap_unchecked();
    let off = (addr - fw.base) as usize;
    *fw.data.get_unchecked(off) as i32
}

#[inline]
#[allow(static_mut_refs)]
unsafe fn firmware_read16(addr: u32) -> i32 {
    let fw = firmware_window.as_ref().unwrap_unchecked();
    let off = (addr - fw.base) as usize;
    ptr::read_unaligned(fw.data.as_ptr().add(off) as *const u16) as i32
}

#[inline]
#[allow(static_mut_refs)]
unsafe fn firmware_read32(addr: u32) -> i32 {
    let fw = firmware_window.as_ref().unwrap_unchecked();
    let off = (addr - fw.base) as usize;
    ptr::read_unaligned(fw.data.as_ptr().add(off) as *const i32)
}

#[inline]
#[allow(static_mut_refs)]
unsafe fn firmware_writable() -> bool {
    firmware_window
        .as_ref()
        .map_or(false, |fw| fw.writable)
}

#[inline]
#[allow(static_mut_refs)]
unsafe fn firmware_write8(addr: u32, value: u8) {
    let fw = firmware_window.as_mut().unwrap_unchecked();
    let off = (addr - fw.base) as usize;
    *fw.data.get_unchecked_mut(off) = value;
}

#[inline]
#[allow(static_mut_refs)]
unsafe fn firmware_write16(addr: u32, value: u16) {
    let fw = firmware_window.as_mut().unwrap_unchecked();
    let off = (addr - fw.base) as usize;
    ptr::write_unaligned(fw.data.as_mut_ptr().add(off) as *mut u16, value);
}

#[inline]
#[allow(static_mut_refs)]
unsafe fn firmware_write32(addr: u32, value: i32) {
    let fw = firmware_window.as_mut().unwrap_unchecked();
    let off = (addr - fw.base) as usize;
    ptr::write_unaligned(fw.data.as_mut_ptr().add(off) as *mut i32, value);
}

#[no_mangle]
pub fn allocate_memory(size: u32) -> u32 {
    unsafe {
        dbg_assert!(mem8.is_null());
    };
    dbg_log!("Allocate memory size={}m", size >> 20);
    let layout = alloc::Layout::from_size_align(size as usize, 0x1000).unwrap();
    let ptr = unsafe { alloc::alloc(layout) as u32 };
    unsafe {
        mem8 = ptr as *mut u8;
    };
    ptr
}

#[no_mangle]
pub unsafe fn zero_memory(addr: u32, size: u32) {
    ptr::write_bytes(mem8.offset(addr as isize), 0, size as usize);
}

#[allow(non_upper_case_globals)]
pub static mut vga_mem8: *mut u8 = ptr::null_mut();
#[allow(non_upper_case_globals)]
pub static mut vga_memory_size: u32 = 0;

#[no_mangle]
pub fn svga_allocate_memory(size: u32) -> u32 {
    unsafe {
        dbg_assert!(vga_mem8.is_null());
    };
    let layout = alloc::Layout::from_size_align(size as usize, 0x1000).unwrap();
    let ptr = unsafe { alloc::alloc(layout) };
    dbg_assert!(
        size & (1 << 12 << 6) == 0,
        "size not aligned to dirty_bitmap"
    );
    unsafe {
        vga_mem8 = ptr;
        vga_memory_size = size;
        vga::set_dirty_bitmap_size(size >> 12 >> 6);
    };
    ptr as u32
}

#[no_mangle]
pub fn in_mapped_range(addr: u32) -> bool {
    return addr >= 0xA0000 && addr < 0xC0000 || addr >= unsafe { *memory_size };
}

pub const VGA_LFB_ADDRESS: u32 = 0xE0000000;
pub fn in_svga_lfb(addr: u32) -> bool {
    addr >= VGA_LFB_ADDRESS && addr <= unsafe { VGA_LFB_ADDRESS + (vga_memory_size - 1) }
}

#[no_mangle]
pub fn read8(addr: u32) -> i32 {
    if in_mapped_range(addr) {
        if in_svga_lfb(addr) {
            unsafe { *vga_mem8.offset((addr - VGA_LFB_ADDRESS) as isize) as i32 }
        }
        else if addr >= APIC_MEM_ADDRESS && addr < APIC_MEM_ADDRESS + APIC_MEM_SIZE {
            apic::read32((addr - APIC_MEM_ADDRESS) & !3) as i32 >> 8 * (addr & 3) & 0xFF
        }
        else if addr >= IOAPIC_MEM_ADDRESS && addr < IOAPIC_MEM_ADDRESS + IOAPIC_MEM_SIZE {
            ioapic::read32((addr - IOAPIC_MEM_ADDRESS) & !3) as i32 >> 8 * (addr & 3) & 0xFF
        }
        else if in_firmware_window(addr) {
            unsafe { firmware_read8(addr) }
        }
        else {
            unsafe { ext::mmap_read8(addr) }
        }
    }
    else {
        read8_no_mmap_check(addr)
    }
}
pub fn read8_no_mmap_check(addr: u32) -> i32 { unsafe { *mem8.offset(addr as isize) as i32 } }

#[no_mangle]
pub fn read16(addr: u32) -> i32 {
    if in_mapped_range(addr) {
        if in_svga_lfb(addr) {
            unsafe {
                ptr::read_unaligned(vga_mem8.offset((addr - VGA_LFB_ADDRESS) as isize) as *const u16)
                    as i32
            }
        }
        else if in_firmware_window(addr) && in_firmware_window(addr + 1) {
            unsafe { firmware_read16(addr) }
        }
        else {
            read8(addr) | read8(addr + 1) << 8
        }
    }
    else {
        read16_no_mmap_check(addr)
    }
}
pub fn read16_no_mmap_check(addr: u32) -> i32 {
    unsafe { ptr::read_unaligned(mem8.offset(addr as isize) as *const u16) as i32 }
}

#[no_mangle]
pub fn read32s(addr: u32) -> i32 {
    if in_mapped_range(addr) {
        if in_svga_lfb(addr) {
            unsafe {
                ptr::read_unaligned(vga_mem8.offset((addr - VGA_LFB_ADDRESS) as isize) as *const i32)
            } // XXX
        }
        else if addr >= APIC_MEM_ADDRESS && addr < APIC_MEM_ADDRESS + APIC_MEM_SIZE {
            apic::read32(addr - APIC_MEM_ADDRESS) as i32
        }
        else if addr >= IOAPIC_MEM_ADDRESS && addr < IOAPIC_MEM_ADDRESS + IOAPIC_MEM_SIZE {
            ioapic::read32(addr - IOAPIC_MEM_ADDRESS) as i32
        }
        else if in_firmware_window(addr) && in_firmware_window(addr + 3) {
            unsafe { firmware_read32(addr) }
        }
        else {
            unsafe { ext::mmap_read32(addr) }
        }
    }
    else {
        read32_no_mmap_check(addr)
    }
}
pub fn read32_no_mmap_check(addr: u32) -> i32 {
    unsafe { ptr::read_unaligned(mem8.offset(addr as isize) as *const i32) }
}

pub unsafe fn read64s(addr: u32) -> i64 {
    if in_mapped_range(addr) {
        if in_svga_lfb(addr) {
            ptr::read_unaligned(vga_mem8.offset((addr - VGA_LFB_ADDRESS) as isize) as *const i64)
        }
        else {
            read32s(addr) as i64 | (read32s(addr + 4) as i64) << 32
        }
    }
    else {
        ptr::read_unaligned(mem8.offset(addr as isize) as *const i64)
    }
}

pub unsafe fn read128(addr: u32) -> reg128 {
    if in_mapped_range(addr) {
        if in_svga_lfb(addr) {
            ptr::read_unaligned(vga_mem8.offset((addr - VGA_LFB_ADDRESS) as isize) as *const reg128)
        }
        else {
            reg128 {
                i32: [
                    read32s(addr + 0),
                    read32s(addr + 4),
                    read32s(addr + 8),
                    read32s(addr + 12),
                ],
            }
        }
    }
    else {
        ptr::read_unaligned(mem8.offset(addr as isize) as *const reg128)
    }
}

#[no_mangle]
pub unsafe fn write8(addr: u32, value: i32) {
    if in_mapped_range(addr) {
        mmap_write8(addr, value & 0xFF);
    }
    else {
        jit::jit_dirty_page(Page::page_of(addr));
        write8_no_mmap_or_dirty_check(addr, value);
    };
}

pub unsafe fn write8_no_mmap_or_dirty_check(addr: u32, value: i32) {
    *mem8.offset(addr as isize) = value as u8
}

#[no_mangle]
pub unsafe fn write16(addr: u32, value: i32) {
    if in_mapped_range(addr) {
        mmap_write16(addr, value & 0xFFFF);
    }
    else {
        jit::jit_dirty_cache_small(addr, addr + 2);
        write16_no_mmap_or_dirty_check(addr, value);
    };
}
pub unsafe fn write16_no_mmap_or_dirty_check(addr: u32, value: i32) {
    ptr::write_unaligned(mem8.offset(addr as isize) as *mut u16, value as u16)
}

#[no_mangle]
pub unsafe fn write32(addr: u32, value: i32) {
    if in_mapped_range(addr) {
        mmap_write32(addr, value);
    }
    else {
        jit::jit_dirty_cache_small(addr, addr + 4);
        write32_no_mmap_or_dirty_check(addr, value);
    }
}

pub unsafe fn write32_no_mmap_or_dirty_check(addr: u32, value: i32) {
    ptr::write_unaligned(mem8.offset(addr as isize) as *mut i32, value)
}

pub unsafe fn write64_no_mmap_or_dirty_check(addr: u32, value: u64) {
    ptr::write_unaligned(mem8.offset(addr as isize) as *mut u64, value)
}

pub unsafe fn write128_no_mmap_or_dirty_check(addr: u32, value: reg128) {
    ptr::write_unaligned(mem8.offset(addr as isize) as *mut reg128, value)
}

pub unsafe fn memset_no_mmap_or_dirty_check(addr: u32, value: u8, count: u32) {
    ptr::write_bytes(mem8.offset(addr as isize), value, count as usize);
}

pub unsafe fn memcpy_no_mmap_or_dirty_check(src_addr: u32, dst_addr: u32, count: u32) {
    dbg_assert!(src_addr < *memory_size);
    dbg_assert!(dst_addr < *memory_size);
    ptr::copy(
        mem8.offset(src_addr as isize),
        mem8.offset(dst_addr as isize),
        count as usize,
    )
}

pub unsafe fn memcpy_into_svga_lfb(src_addr: u32, dst_addr: u32, count: u32) {
    dbg_assert!(src_addr < *memory_size);
    dbg_assert!(in_svga_lfb(dst_addr));
    dbg_assert!(Page::page_of(dst_addr) == Page::page_of(dst_addr + count - 1));
    vga::mark_dirty(dst_addr);
    ptr::copy_nonoverlapping(
        mem8.offset(src_addr as isize),
        vga_mem8.offset((dst_addr - VGA_LFB_ADDRESS) as isize),
        count as usize,
    )
}

pub unsafe fn mmap_write8(addr: u32, value: i32) {
    if in_svga_lfb(addr) {
        vga::mark_dirty(addr);
        *vga_mem8.offset((addr - VGA_LFB_ADDRESS) as isize) = value as u8
    }
    else if in_firmware_window(addr) {
        if firmware_writable() {
            firmware_write8(addr, value as u8);
        }
    }
    else {
        ext::mmap_write8(addr, value)
    }
}
pub unsafe fn mmap_write16(addr: u32, value: i32) {
    if in_svga_lfb(addr) {
        vga::mark_dirty(addr);
        ptr::write_unaligned(
            vga_mem8.offset((addr - VGA_LFB_ADDRESS) as isize) as *mut u16,
            value as u16,
        )
    }
    else if in_firmware_window(addr) {
        if firmware_writable() {
            firmware_write16(addr, value as u16);
        }
    }
    else {
        ext::mmap_write16(addr, value)
    }
}
pub unsafe fn mmap_write32(addr: u32, value: i32) {
    if in_svga_lfb(addr) {
        vga::mark_dirty(addr);
        ptr::write_unaligned(
            vga_mem8.offset((addr - VGA_LFB_ADDRESS) as isize) as *mut i32,
            value,
        )
    }
    else if addr >= APIC_MEM_ADDRESS && addr < APIC_MEM_ADDRESS + APIC_MEM_SIZE {
        apic::write32(addr - APIC_MEM_ADDRESS, value as u32);
        handle_irqs();
    }
    else if addr >= IOAPIC_MEM_ADDRESS && addr < IOAPIC_MEM_ADDRESS + IOAPIC_MEM_SIZE {
        ioapic::write32(addr - IOAPIC_MEM_ADDRESS, value as u32);
        handle_irqs();
    }
    else if in_firmware_window(addr) {
        if firmware_writable() {
            firmware_write32(addr, value);
        }
    }
    else {
        ext::mmap_write32(addr, value)
    }
}
pub unsafe fn mmap_write64(addr: u32, value: u64) {
    if in_svga_lfb(addr) {
        vga::mark_dirty(addr);
        ptr::write_unaligned(
            vga_mem8.offset((addr - VGA_LFB_ADDRESS) as isize) as *mut u64,
            value,
        )
    }
    else {
        ext::mmap_write64(addr, value as i32, (value >> 32) as i32)
    }
}
pub unsafe fn mmap_write128(addr: u32, v0: u64, v1: u64) {
    if in_svga_lfb(addr) {
        vga::mark_dirty(addr);
        ptr::write_unaligned(
            vga_mem8.offset((addr - VGA_LFB_ADDRESS) as isize) as *mut u64,
            v0,
        );
        ptr::write_unaligned(
            vga_mem8.offset((addr - VGA_LFB_ADDRESS + 8) as isize) as *mut u64,
            v1,
        )
    }
    else {
        ext::mmap_write128(
            addr,
            v0 as i32,
            (v0 >> 32) as i32,
            v1 as i32,
            (v1 >> 32) as i32,
        )
    }
}

#[no_mangle]
pub unsafe fn is_memory_zeroed(addr: u32, length: u32) -> bool {
    dbg_assert!(addr % 8 == 0);
    dbg_assert!(length % 8 == 0);
    for i in (addr..addr + length).step_by(8) {
        if *(mem8.offset(i as isize) as *const i64) != 0 {
            return false;
        }
    }
    return true;
}
