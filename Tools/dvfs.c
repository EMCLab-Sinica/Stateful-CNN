#include "driverlib.h"
#include "Tools/dvfs.h"

#ifdef __MSP430__
unsigned int FreqLevel = 8;
#elif __MSP432__
unsigned int FreqLevel = 1;
#endif

/*
 * 8 level of CPU frequency for MSP430
 *
 * Level: 1: 1MHz
 *        2: 2.67MHz
 *        3: 3.33MHz
 *        4: 4MHz
 *        5: 5.33MHz
 *        6: 6.67MHz
 *        7: 8MHz
 *        8: 16MHz
 * 
 * 1 level of CPU frequency for MSP432
 * Level: 1: 12MHz
 */
void setFrequency(int level)
{
    switch(level)
    {
#ifdef __MSP430__
    case 1:// Set DCO frequency to 1 MHz
        CS_setDCOFreq(CS_DCORSEL_0, CS_DCOFSEL_0);
        FreqLevel = 1;
        break;
    case 2:// Set DCO frequency to 2.67 MHz
        CS_setDCOFreq(CS_DCORSEL_0, CS_DCOFSEL_1);
        FreqLevel = 2;
        break;
    case 3:// Set DCO frequency to 3.5 MHz
        CS_setDCOFreq(CS_DCORSEL_0, CS_DCOFSEL_2);
        FreqLevel = 3;
        break;
    case 4:// Set DCO frequency to 4 MHz
        CS_setDCOFreq(CS_DCORSEL_0, CS_DCOFSEL_3);
        FreqLevel = 4;
        break;
    case 5:// Set DCO frequency to 5.33 MHz
        CS_setDCOFreq(CS_DCORSEL_0, CS_DCOFSEL_4);
        FreqLevel = 5;
        break;
    case 6:// Set DCO frequency to 7 MHz
        CS_setDCOFreq(CS_DCORSEL_0, CS_DCOFSEL_5);
        FreqLevel = 6;
        break;
    case 7:// Set DCO frequency to 8 MHz
        CS_setDCOFreq(CS_DCORSEL_0, CS_DCOFSEL_6);
        FreqLevel = 7;
        break;
    case 8:// Set DCO frequency to 16 MHz
        FRCTL0 = FRCTLPW | NWAITS_1; //Up to 16Mhz
        CS_setDCOFreq(CS_DCORSEL_1, CS_DCOFSEL_4);
        FreqLevel = 8;
        break;
#elif defined(__MSP432__)
    case 1: // Set DCO to 12MHz
        CS_setDCOCenteredFrequency(CS_DCO_FREQUENCY_12);
#endif
    default:
        break;
    }
}

unsigned long getFrequency(int level)
{
    switch(level)
    {
#ifdef __MSP430__
        case 1:// Set DCO frequency to 1 MHz
            return 1000000;
        case 2:// Set DCO frequency to 2.67 MHz
            return 2670000;
        case 3:// Set DCO frequency to 3.33 MHz
            return 3330000;
        case 4:// Set DCO frequency to 4 MHz
            return 4000000;
        case 5:// Set DCO frequency to 5.33 MHz
            return 5330000;
        case 6:// Set DCO frequency to 6.67 MHz
            return 6670000;
        case 7:// Set DCO frequency to 8 MHz
            return 8000000;
        case 8:// Set DCO frequency to 16 MHz
            return 16000000;
#elif defined(__MSP432__)
        case 1: // Set DCO frequency to 12MHz
            return 12000000;
#endif
        default:
            break;
    }
    return 0;
}
