#pragma once
// Read-only file mapping for checkpoint shards: mmap on POSIX, a file mapping on Windows.
#include <cstddef>

#ifdef _WIN32
#define BOLT_API __declspec(dllexport)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// The whole file mapped read-only, its length in *size; nullptr when it cannot be.
static inline void* bolt_map_file(const char* path, size_t* size) {
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return nullptr;
    LARGE_INTEGER n;
    if (!GetFileSizeEx(f, &n) || n.QuadPart == 0) { CloseHandle(f); return nullptr; }
    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(f);
    if (!m) return nullptr;
    void* p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m);  // the view keeps the mapping alive
    if (!p) return nullptr;
    *size = (size_t)n.QuadPart;
    return p;
}
static inline void bolt_will_need(void* p, size_t size) {
    WIN32_MEMORY_RANGE_ENTRY r{p, size};
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &r, 0);
}
static inline void bolt_unmap_file(void* p, size_t) { UnmapViewOfFile(p); }
#else
#define BOLT_API __attribute__((visibility("default")))
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static inline void* bolt_map_file(const char* path, size_t* size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return nullptr; }
    void* p = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return nullptr;
    *size = (size_t)st.st_size;
    return p;
}
static inline void bolt_will_need(void* p, size_t size) { madvise(p, size, MADV_WILLNEED); }
static inline void bolt_unmap_file(void* p, size_t size) { munmap(p, size); }
#endif
