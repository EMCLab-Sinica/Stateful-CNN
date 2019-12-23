/*
 *  main.c
 *
 *  Author: Meenchen
 */
#define TestDB

/* Scheduler include files. */
#include <RecoveryHandler/Recovery.h>
#include <Tools/myuart.h>
#include <Tools/dvfs.h>
#include <FreeRTOS.h>
#include <task.h>
#include <semphr.h>

#ifdef TestStack
#include "StackInVM.h"
#endif
#ifdef TestDB
#include <DBTest.h>
#endif
#ifdef TestFail
#include <FailureTest.h>
#endif

#include "ExpTests/IntermittentCNNTest.h"

/* Standard demo includes, used so the tick hook can exercise some FreeRTOS
functionality in an interrupt. */
#include <driverlib.h>
#include <main.h>

/*-----------------------------------------------------------*/
/*
 * Configure the hardware as necessary.
 */
static void prvSetupHardware( void );

/*-----------------------------------------------------------*/

int main( void )
{
    int i;
    //ADC's resource
    ADCSemph = 0;

    /* Configure the hardware ready to run the demo. */
    prvSetupHardware();

	if(firstTime != 1){
	    capID = -1;
	    tempID = -1;
	    avgtempID = -1;
	    avgcapID = -1;
	    timeCounter = 0;
	    pvInitHeapVar();
	    for(i = 0; i < NUMTASK;i++)
	        information[i] = 0;
        //main_DBtest();
        IntermittentCNNTest();
	}
	else{
	    failureRecovery();
	}
	return 0;
}


static void prvSetupHardware( void )
{
    /* Stop Watchdog timer. */
    WDT_A_hold( __MSP430_BASEADDRESS_WDT_A__ );

	/* Set all GPIO pins to output and low. */
	GPIO_setOutputLowOnPin( GPIO_PORT_P1, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 );
	GPIO_setOutputLowOnPin( GPIO_PORT_P2, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 );
	GPIO_setOutputLowOnPin( GPIO_PORT_P3, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 );
	GPIO_setOutputLowOnPin( GPIO_PORT_P4, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 );
	GPIO_setOutputLowOnPin( GPIO_PORT_PJ, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 | GPIO_PIN8 | GPIO_PIN9 | GPIO_PIN10 | GPIO_PIN11 | GPIO_PIN12 | GPIO_PIN13 | GPIO_PIN14 | GPIO_PIN15 );
	GPIO_setAsOutputPin( GPIO_PORT_P1, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 );
	GPIO_setAsOutputPin( GPIO_PORT_P2, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 );
	GPIO_setAsOutputPin( GPIO_PORT_P3, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 );
	GPIO_setAsOutputPin( GPIO_PORT_P4, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 );
	GPIO_setAsOutputPin( GPIO_PORT_PJ, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 | GPIO_PIN8 | GPIO_PIN9 | GPIO_PIN10 | GPIO_PIN11 | GPIO_PIN12 | GPIO_PIN13 | GPIO_PIN14 | GPIO_PIN15 );

	/* Configure P2.0 - UCA0TXD and P2.1 - UCA0RXD. */
	GPIO_setOutputLowOnPin( GPIO_PORT_P2, GPIO_PIN0 );
	GPIO_setAsOutputPin( GPIO_PORT_P2, GPIO_PIN0 );
	GPIO_setAsPeripheralModuleFunctionInputPin( GPIO_PORT_P2, GPIO_PIN1, GPIO_SECONDARY_MODULE_FUNCTION );
	GPIO_setAsPeripheralModuleFunctionOutputPin( GPIO_PORT_P2, GPIO_PIN0, GPIO_SECONDARY_MODULE_FUNCTION );

	/* Set PJ.4 and PJ.5 for LFXT. */
	GPIO_setAsPeripheralModuleFunctionInputPin(  GPIO_PORT_PJ, GPIO_PIN4 + GPIO_PIN5, GPIO_PRIMARY_MODULE_FUNCTION  );

	// Configure button S1 (P5.6) interrupt and S2 P(5.5)
    GPIO_selectInterruptEdge(GPIO_PORT_P5, GPIO_PIN6, GPIO_HIGH_TO_LOW_TRANSITION);
    GPIO_setAsInputPinWithPullUpResistor(GPIO_PORT_P5, GPIO_PIN6);
    GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN6);
    GPIO_enableInterrupt(GPIO_PORT_P5, GPIO_PIN6);
    GPIO_selectInterruptEdge(GPIO_PORT_P5, GPIO_PIN5, GPIO_HIGH_TO_LOW_TRANSITION);
    GPIO_setAsInputPinWithPullUpResistor(GPIO_PORT_P5, GPIO_PIN5);
    GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN5);
    GPIO_enableInterrupt(GPIO_PORT_P5, GPIO_PIN5);

    GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN6);
    GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN5);

	/* Set DCO frequency to 16 MHz. */
	setFrequency(FreqLevel);

	/* Set external clock frequency to 32.768 KHz. */
	CS_setExternalClockSource( 32768, 0 );

	/* Set ACLK = LFXT. */
	CS_initClockSignal( CS_ACLK, CS_LFXTCLK_SELECT, CS_CLOCK_DIVIDER_1 );

	/* Set SMCLK = DCO with frequency divider of 1. */
	CS_initClockSignal( CS_SMCLK, CS_DCOCLK_SELECT, CS_CLOCK_DIVIDER_1 );

	/* Set MCLK = DCO with frequency divider of 1. */
	CS_initClockSignal( CS_MCLK, CS_DCOCLK_SELECT, CS_CLOCK_DIVIDER_1 );

	/* Start XT1 with no time out. */
	CS_turnOnLFXT( CS_LFXT_DRIVE_0 );

	/* Disable the GPIO power-on default high-impedance mode. */
	PMM_unlockLPM5();

	/* Initail Uart */
	uartinit();
}
/*-----------------------------------------------------------*/

int _system_pre_init( void )
{
    /* Stop Watchdog timer. */
    WDT_A_hold( __MSP430_BASEADDRESS_WDT_A__ );

    /* Return 1 for segments to be initialised. */
    return 1;
}

/*
 * port 5 interrupt service routine to handle s1 and s2 button press
 *
 */
#pragma vector=PORT5_VECTOR
__interrupt void Port_5(void)
{
    GPIO_disableInterrupt(GPIO_PORT_P5, GPIO_PIN6);
    GPIO_disableInterrupt(GPIO_PORT_P5, GPIO_PIN5);

    /* Button pushed, do something if you need to */

    GPIO_enableInterrupt(GPIO_PORT_P5, GPIO_PIN6);
    GPIO_enableInterrupt(GPIO_PORT_P5, GPIO_PIN5);
    GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN6);
    GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN5);
}


/* Temperature and voltage
 * ADC12 Interrupt Service Routine
 * Exits LPM3 when Temperature/Voltage data is ready
 */
#pragma vector = ADC12_VECTOR
__interrupt void ADC12_ISR(void)
{
  switch(__even_in_range(ADC12IV,76))
  {
    case  ADC12IV_NONE: break;                // Vector  0:  No interrupt
    case  ADC12IV_ADC12OVIFG: break;          // Vector  2:  ADC12MEMx Overflow
    case  ADC12IV_ADC12TOVIFG: break;         // Vector  4:  Conversion time overflow
    case  ADC12IV_ADC12HIIFG: break;          // Vector  6:  ADC12HI
    case  ADC12IV_ADC12LOIFG: break;          // Vector  8:  ADC12LO
    case ADC12IV_ADC12INIFG: break;           // Vector 10:  ADC12IN
    case ADC12IV_ADC12IFG0:                   // Vector 12:  ADC12MEM0
        ADC12IFGR0 &= ~ADC12IFG0;             // Clear interrupt flag
        waitCap = 0;
        __bic_SR_register_on_exit(LPM3_bits); // Exit active CPU
        break;
    case ADC12IV_ADC12IFG1:                   // Vector 14:  ADC12MEM1
        break;
    case ADC12IV_ADC12IFG2:
        ADC12IFGR0 &= ~ADC12IFG2;
        waitTemp = 0;
        __bic_SR_register_on_exit(LPM3_bits); // Exit active CPU
        break;            // Vector 16:  ADC12MEM2
    case ADC12IV_ADC12IFG3: break;            // Vector 18:  ADC12MEM3
    case ADC12IV_ADC12IFG4: break;            // Vector 20:  ADC12MEM4
    case ADC12IV_ADC12IFG5: break;            // Vector 22:  ADC12MEM5
    case ADC12IV_ADC12IFG6: break;            // Vector 24:  ADC12MEM6
    case ADC12IV_ADC12IFG7: break;            // Vector 26:  ADC12MEM7
    case ADC12IV_ADC12IFG8: break;            // Vector 28:  ADC12MEM8
    case ADC12IV_ADC12IFG9: break;            // Vector 30:  ADC12MEM9
    case ADC12IV_ADC12IFG10: break;           // Vector 32:  ADC12MEM10
    case ADC12IV_ADC12IFG11: break;           // Vector 34:  ADC12MEM11
    case ADC12IV_ADC12IFG12: break;           // Vector 36:  ADC12MEM12
    case ADC12IV_ADC12IFG13: break;           // Vector 38:  ADC12MEM13
    case ADC12IV_ADC12IFG14: break;           // Vector 40:  ADC12MEM14
    case ADC12IV_ADC12IFG15: break;           // Vector 42:  ADC12MEM15
    case ADC12IV_ADC12IFG16: break;           // Vector 44:  ADC12MEM16
    case ADC12IV_ADC12IFG17: break;           // Vector 46:  ADC12MEM17
    case ADC12IV_ADC12IFG18: break;           // Vector 48:  ADC12MEM18
    case ADC12IV_ADC12IFG19: break;           // Vector 50:  ADC12MEM19
    case ADC12IV_ADC12IFG20: break;           // Vector 52:  ADC12MEM20
    case ADC12IV_ADC12IFG21: break;           // Vector 54:  ADC12MEM21
    case ADC12IV_ADC12IFG22: break;           // Vector 56:  ADC12MEM22
    case ADC12IV_ADC12IFG23: break;           // Vector 58:  ADC12MEM23
    case ADC12IV_ADC12IFG24: break;           // Vector 60:  ADC12MEM24
    case ADC12IV_ADC12IFG25: break;           // Vector 62:  ADC12MEM25
    case ADC12IV_ADC12IFG26: break;           // Vector 64:  ADC12MEM26
    case ADC12IV_ADC12IFG27: break;           // Vector 66:  ADC12MEM27
    case ADC12IV_ADC12IFG28: break;           // Vector 68:  ADC12MEM28
    case ADC12IV_ADC12IFG29: break;           // Vector 70:  ADC12MEM29
    case ADC12IV_ADC12IFG30: break;           // Vector 72:  ADC12MEM30
    case ADC12IV_ADC12IFG31: break;           // Vector 74:  ADC12MEM31
    case ADC12IV_ADC12RDYIFG: break;          // Vector 76:  ADC12RDY
    default: break;
  }
}
