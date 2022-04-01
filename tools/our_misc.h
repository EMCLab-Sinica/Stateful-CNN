#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __arm__
inline void our_delay_cycles_internal(uint32_t n_cycles) {
    // Based on an example in https://community.arm.com/developer/ip-products/processors/b/processors-ip-blog/posts/condition-codes-1-condition-flags-and-codes
    // %= guarantees unique labels. See https://www.keil.com/support/man/docs/armclang_ref/armclang_ref_wan1517569524985.htm
    __asm__ volatile(".Lour_delay_cycles_loop%=:\n"
                     "nop\n"
                     "subs %[n_cycles], %[n_cycles], #1\n"
                     "bne .Lour_delay_cycles_loop%=\n"
                     :
                     : [n_cycles] "r"(n_cycles));
}
#elif defined(__MSP430__)
# ifdef __GNUC__
inline void our_delay_cycles_internal(uint32_t n_cycles) {
    __asm__ volatile(".Lour_delay_cycles_loop%=:\n"
                     "NOP\n"
                     "DEC %[n_cycles]\n"
                     "JNE .Lour_delay_cycles_loop%=\n"
                     :
                     : [n_cycles] "r"(n_cycles));
}
# else
void our_delay_cycles_internal(uint16_t n_cycles);
#endif
#else
#define our_delay_cycles_internal(n_cycles)
#endif

// both subs and nop take 1 cycle, bne takes 1 + P cycles, and MSP430/MSP432 uses a 3-stage pipeline
// the additional nop instruction makes calculating the number of iterations faster - division by
// 3 involves a slow loop on MSP430 and a slow instruction on other platforms
#define our_delay_cycles(n_cycles) our_delay_cycles_internal(n_cycles / 4)

#ifdef __cplusplus
}
#endif
