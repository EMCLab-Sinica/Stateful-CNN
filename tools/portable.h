#pragma once

#if defined(__MSP430__) || defined(__MSP432__)
#define __TOOLS_MSP__
#endif

#ifdef __STM32__

#ifdef STM32L496xx
#define STM32_HAL_HEADER "stm32l4xx_hal.h"
#else
#error "Please verify and add corresponding macros and headers"
#endif

#endif
