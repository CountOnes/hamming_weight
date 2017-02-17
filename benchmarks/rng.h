#ifndef RNG_H

#include <stdlib.h>
#include <stdint.h>

static uint32_t get32rand() {
    return (((uint32_t)rand() << 0) & 0x0000FFFFul) |
           (((uint32_t)rand() << 16) & 0xFFFF0000ul);
}

static uint64_t get64rand() {
    return (((uint64_t)get32rand()) << 32) | get32rand();
}
#endif
