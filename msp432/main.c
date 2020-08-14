#include <driverlib.h>
#include "msp.h"
#include "plat-msp430.h"
#include "Tools/myuart.h"
#include "Tools/dvfs.h"

/**
 * main.c
 */

void main(void)
{
	WDT_A->CTL = WDT_A_CTL_PW | WDT_A_CTL_HOLD;		// stop watchdog timer

    setFrequency(FreqLevel);
    uartinit();

    IntermittentCNNTest();
}
