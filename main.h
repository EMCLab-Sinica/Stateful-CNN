/*
 *  main.h
 *
 *  Author: Meenchen
 */

#define configTICK_VECTOR           TIMER0_A0_VECTOR
#define configTICK_RATE_HZ          ( 1000 ) /* In this non-real time simulated environment the tick frequency has to be at least a multiple of the Win32 tick frequency, and therefore very slow. */
