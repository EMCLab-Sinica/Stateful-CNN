#include <driverlib.h>
#include "msp.h"
#include "plat-mcu.h"
#include "tools/myuart.h"
#include "tools/dvfs.h"

/**
 * main.c
 */

static void prvSetupHardware( void );

void main(void)
{
	WDT_A->CTL = WDT_A_CTL_PW | WDT_A_CTL_HOLD;		// stop watchdog timer

    setFrequency(FreqLevel);

    prvSetupHardware();

    IntermittentCNNTest();
}

// See timer_a_upmode_gpio_toggle.c in MSP432 examples for code below

#define TIMER_PERIOD    375

static void prvSetupHardware( void ) {
    // Ref: MSP432 example gpio_input_interrupt.c

    /* Configuring P1.1 as an input and enabling interrupts */
    MAP_GPIO_setAsInputPinWithPullUpResistor(GPIO_PORT_P1, GPIO_PIN1|GPIO_PIN4);
    MAP_GPIO_clearInterruptFlag(GPIO_PORT_P1, GPIO_PIN1|GPIO_PIN4);
    MAP_GPIO_enableInterrupt(GPIO_PORT_P1, GPIO_PIN1|GPIO_PIN4);
    MAP_Interrupt_enableInterrupt(INT_PORT1);

    /* Enabling MASTER interrupts */
    MAP_Interrupt_enableMaster();
}

/* GPIO ISR */
void PORT1_IRQHandler(void)
{
    uint32_t status;

    status = MAP_GPIO_getEnabledInterruptStatus(GPIO_PORT_P1);
    MAP_GPIO_clearInterruptFlag(GPIO_PORT_P1, status);

    button_pushed(status & GPIO_PIN1, status & GPIO_PIN4);
}
