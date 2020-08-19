#include <driverlib.h>
#include "msp.h"
#include "plat-msp430.h"
#include "Tools/myuart.h"
#include "Tools/dvfs.h"

/**
 * main.c
 */

void timerinit(void);

void main(void)
{
	WDT_A->CTL = WDT_A_CTL_PW | WDT_A_CTL_HOLD;		// stop watchdog timer

    setFrequency(FreqLevel);
    uartinit();
    timerinit();

    IntermittentCNNTest();
}

// See timer_a_upmode_gpio_toggle.c in MSP432 examples for code below

#define TIMER_PERIOD    375

/* Timer_A UpMode Configuration Parameter */
static const Timer_A_UpModeConfig upConfig = {
    .clockSource = TIMER_A_CLOCKSOURCE_SMCLK,
    .clockSourceDivider = TIMER_A_CLOCKSOURCE_DIVIDER_32,
    // SMCLK is 12MHz (from CS_getSMCLK()), so 1ms has 12M / 32 / 1000 = 375 ticks
    .timerPeriod = TIMER_PERIOD,
    .timerInterruptEnable_TAIE = TIMER_A_TAIE_INTERRUPT_DISABLE,
    .captureCompareInterruptEnable_CCR0_CCIE = TIMER_A_CCIE_CCR0_INTERRUPT_ENABLE,
    .timerClear = TIMER_A_DO_CLEAR
};

void timerinit(void) {
    /* Configuring Timer_A1 for Up Mode */
    MAP_Timer_A_configureUpMode(TIMER_A1_BASE, &upConfig);

    /* Enabling interrupts and starting the timer */
    MAP_Interrupt_enableInterrupt(INT_TA1_0);
    MAP_Timer_A_startCounter(TIMER_A1_BASE, TIMER_A_UP_MODE);

    /* Enabling MASTER interrupts */
    MAP_Interrupt_enableMaster();
}
