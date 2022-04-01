/*
 * extfram.h
 *
 *  Created on: 2020年7月8日
 *  Range : 0x00000 - 0xFFFFF (20bit) //8Mb
 *          0x00000 - 0x7FFFF // 4Mb
 *  Clock : up to 20 MHz (8Mb)
 *          up to 40 MHz (4Mb)
 */
//                   MSP430FR5994
//                 -----------------
//                |             P1.0|-> Comms LED
//                |                 |
//                |                 |
//                |                 |
//                |             P6.0|-> Data Out (UCA3SIMO)
//                |                 |
//                |             P6.1|<- Data In (UCA3SOMI)
//                |                 |
//                |             P6.2|-> Serial Clock Out (UCA3	CLK)
//                |                 |
//                |             P6.3|-> Slave Chip Select (GPIO)


//                   MSP432P401R (100 pin, using UCA1)
//                 -----------------
//                |             P1.0|-> Comms LED
//                |                 |
//                |                 |
//                |                 |
//                |             P2.3|-> Data Out (UCA1SIMO <-> SDI)
//                |                 |
//                |             P2.2|<- Data In (UCA1SOMI <-> SDO)
//                |                 |
//                |             P2.1|-> Serial Clock Out (UCA1CLK <-> SCK)
//                |                 |
//                |             P1.5|-> Slave Chip Select (GPIO <-> CS)

#include "extfram.h"
#include <string.h>
#include <stdint.h>

#if defined(__MSP430__) || defined(__MSP432__)
#define __EXT_FRAM_MSP__
#endif

#ifdef __EXT_FRAM_MSP__
#include <driverlib.h>

#ifdef __MSP430__
#include <msp430.h>
#elif defined(__MSP432__)
#include <msp.h>
uint8_t controlTable[1024];
uint32_t curDMATransmitChannelNum, curDMAReceiveChannelNum;
#endif

static uint16_t dma_timer_delay;

#ifdef __MSP430__
#define UCA3
#else
#define UCA1
#endif

#ifdef UCA3

#define DMA3TSEL__SPITXIFG DMA3TSEL__UCA3TXIFG
#define DMA4TSEL__SPIRXIFG DMA4TSEL__UCA3RXIFG
#define SPITXBUF UCA3TXBUF
#define SPIRXBUF UCA3RXBUF
#define SPISTATW UCA3STATW
#define SPICTLW0 UCA3CTLW0
#define SPIIFG UCA3IFG
#define SPIBRW UCA3BRW
#define SPISEL0 P6SEL0
#define SPISEL1 P6SEL1
#define MSP432_DMA_EUSCI_TRANSMIT_CHANNEL DMA_CH6_EUSCIA3TX
#define MSP432_DMA_EUSCI_RECEIVE_CHANNEL DMA_CH7_EUSCIA3RX
#define MSP432_DMA_EUSCI_MODULE EUSCI_A3_BASE
#endif

#ifdef UCB1
//Does not support DMA read,
#define DMA3TSEL__SPITXIFG DMA3TSEL__UCB1TXIFG
#define SPITXBUF UCB1TXBUF
#define SPIRXBUF UCB1RXBUF
#define SPISTATW UCB1STATW
#define SPICTLW0 UCB1CTLW0
#define SPIIFG UCB1IFG
#define SPIBRW UCB1BRW
#define SPISEL0 P5SEL0
#define SPISEL1 P5SEL1
#define MSP432_DMA_EUSCI_TRANSMIT_CHANNEL DMA_CH2_EUSCIB1TX0
#define MSP432_DMA_EUSCI_RECEIVE_CHANNEL DMA_CH3_EUSCIB1RX0
#define MSP432_DMA_EUSCI_MODULE EUSCI_B1_BASE
#endif

#ifdef UCA1
#define DMA3TSEL__SPITXIFG DMA3TSEL__UCA1TXIFG
#define SPITXBUF UCA1TXBUF
#define SPIRXBUF UCA1RXBUF
#define SPISTATW UCA1STATW
#define SPICTLW0 UCA1CTLW0
#define SPIIFG UCA1IFG
#define SPIBRW UCA1BRW
#define SPISEL0 P5SEL0
#define SPISEL1 P5SEL1
#define MSP432_DMA_EUSCI_TRANSMIT_CHANNEL DMA_CH2_EUSCIA1TX
#define MSP432_DMA_EUSCI_RECEIVE_CHANNEL DMA_CH3_EUSCIA1RX
#define MSP432_DMA_EUSCI_MODULE EUSCI_A1_BASE
#endif

#define MSP432_DMA_EUSCI_TRANSMIT_CHANNEL_NUM (MSP432_DMA_EUSCI_TRANSMIT_CHANNEL & 0x0F)
#define MSP432_DMA_EUSCI_RECEIVE_CHANNEL_NUM (MSP432_DMA_EUSCI_RECEIVE_CHANNEL & 0x0F)

#ifdef __MSP430__
#define SLAVE_CS_OUT    P6OUT
#define SLAVE_CS_DIR    P6DIR
#define SLAVE_CS_PIN    BIT3
#else
#define SLAVE_CS_OUT    P1OUT
#define SLAVE_CS_DIR    P1DIR
#define SLAVE_CS_PIN    BIT5
#endif


#define COMMS_LED_OUT   P1OUT
#define COMMS_LED_DIR   P1DIR
#define COMMS_LED_PIN   BIT0

//WREN Set write enable latch 0000 0110b
#define CMD_WREN 0x06
//WRDI Reset write enable latch 0000 0100b
#define CMD_WRDI 0x04
//RDSR Read Status Register 0000 0101b
#define CMD_RDSR 0x05
//WRSR Write Status Register 0000 0001b
#define CMD_WRSR 0x01
//READ Read memory data 0000 0011b
#define CMD_READ 0x03
//FSTRD Fast read memory data 0000 1011b
#define CMD_FSTRD 0x0b
//WRITE Write memory data 0000 0010b
#define CMD_WRITE 0x02
//SLEEP Enter sleep mode 1011 1001b
#define CMD_SLEEP 0xb9
//RDID Read device ID 1001 1111b
#define CMD_RDID 0x9f
//SSWR Special Sector Write 42h 0100 0010b
#define CMD_SSWR 0x42
//SSRD Special Sector Read 4Bh 0100 1011b
#define CMD_SSRD 0x4b
//RUID Read Unique ID 4Ch 0100 1100b
#define CMD_RUID 0x4c
//DPD Enter Deep Power-Down BAh 1011 1010b
#define CMD_DPD 0xba
//HBN Enter Hibernate mode B9h 1011 1001b
#define CMD_HBN 0xb9

#define DUMMY   0xFF

#define MIN_VAL(x, y) ((x) < (y) ? (x) : (y))

#elif defined(__STM32__)

#include "qspiFRAM.h"

#else

#error "Please defined __MSP430__, __MSP432__ or __STM32__ according to the target board"

#endif

void eraseFRAM() {
	eraseFRAM2(0xff);
}

void eraseFRAM2(uint8_t init_val){
	uint8_t val;
#ifdef FRAM_8Mb
	unsigned long cnt = 0xfffff;
#else
	unsigned long cnt = 0x7ffff;
#endif

#ifdef __EXT_FRAM_MSP__
	SLAVE_CS_OUT &= ~(SLAVE_CS_PIN);
		SPITXBUF = CMD_WREN;
		while(SPISTATW & 0x1);
	SLAVE_CS_OUT |= SLAVE_CS_PIN;

    SLAVE_CS_OUT &= ~(SLAVE_CS_PIN);
    	SPITXBUF = CMD_WRITE;
    	while(SPISTATW & 0x1);
    	SPITXBUF=0x00;
    	while(SPISTATW & 0x1);
    	SPITXBUF=0x00;
    	while(SPISTATW & 0x1);
    	SPITXBUF=0x00;
		while(cnt--){
			while(SPISTATW & 0x1);
			SPITXBUF = init_val;
		}
		COMMS_LED_OUT |=0x1;
		val = SPIRXBUF; //Clean the overrun flag
	SLAVE_CS_OUT |= SLAVE_CS_PIN;
#elif defined(__STM32__)
	uint8_t buffer[64];
	for (uint8_t idx = 0; idx < 64; idx++) {
		buffer[idx] = init_val;
	}
	for (uint32_t addr = 0; addr < cnt; addr += 64) {
		FRAM_Write(addr, buffer, 64);
	}
#endif
}

void initSPI()
{
#ifdef __EXT_FRAM_MSP__
#ifdef __MSP430__
    SPISEL0 |= 0x07;
    SPISEL1 &= 0xF8;
#else
#ifdef UCA3
    // XXX: Not tested yet
    P9SEL0 |= BIT5 | BIT6 | BIT7;
    P9SEL1 &= ~(BIT5 | BIT6 | BIT7);
#endif
#ifdef UCB1
    P6SEL0 |= BIT3 | BIT4 | BIT5;
    P6SEL1 &= ~(BIT3 | BIT4 | BIT5);
#endif
#ifdef UCA1
    P2SEL0 |= BIT1 | BIT2 | BIT3;
    P2SEL1 &= ~(BIT1 | BIT2 | BIT3);
#endif
#endif
    SLAVE_CS_DIR |= SLAVE_CS_PIN;
    SLAVE_CS_OUT |= SLAVE_CS_PIN;

    //Clock Polarity: The inactive state is high
    //MSB First, 8-bit, Master, 3-pin mode, Synchronous
    SPICTLW0 = UCSWRST;                       // **Put state machine in reset**
    SPICTLW0 |= UCCKPH | UCMSB | UCSYNC
                       | UCMST | UCSSEL__SMCLK;      // 3-pin, 8-bit SPI Slave

    //FRAM SPEED control
#ifdef __MSP430__
    SPIBRW = 4;
#else
    // Somehow too fast SPI clocks result in incorrect values in SPI_READ with DMA
    SPIBRW = 4;
#endif

    SPICTLW0 &= ~UCSWRST;                     // **Initialize USCI state machine**

    //All writes to the memory begin with a WREN opcode with CS being asserted and deasserted.
	SLAVE_CS_OUT &= ~(SLAVE_CS_PIN);
		SPITXBUF = CMD_WREN;
		while(SPISTATW & 0x1);
	SLAVE_CS_OUT |= SLAVE_CS_PIN;

	//Disable memory write protection (Just in case)
	SLAVE_CS_OUT &= ~(SLAVE_CS_PIN);
		SPITXBUF = CMD_WRSR;
		while(SPISTATW & 0x1);
		SPITXBUF = 0xC0;
		while(SPISTATW & 0x1);
	SLAVE_CS_OUT |= SLAVE_CS_PIN;
#elif defined(__STM32__)
	FRAM_Interface_Reset();
	FRAM_WREN();
#endif
}
void SPI_READ(SPI_ADDR* A,uint8_t *dst, unsigned long len ){
#ifdef __EXT_FRAM_MSP__
	uint8_t dummy = 0x00;

	SLAVE_CS_OUT &= ~(SLAVE_CS_PIN);
		SPITXBUF = CMD_READ;
		while(SPISTATW & 0x1); //SPI BUSY
		SPITXBUF=A->byte[2];
		while(SPISTATW & 0x1);
		SPITXBUF=A->byte[1];
		while(SPISTATW & 0x1);
		SPITXBUF=A->byte[0];
		while(SPISTATW & 0x1);
#ifdef __MSP430__

		DMACTL1 = (DMACTL1 & 0x00ff) | DMA3TSEL__SPITXIFG;
		// Write dummy data to TX
		DMA3CTL = DMADT_0 + DMADSTINCR_0 + DMASRCINCR_0 +  DMADSTBYTE__BYTE  + DMASRCBYTE__BYTE + DMALEVEL__EDGE;
		DMA3SA = &SPIRXBUF;
		DMA3DA = &SPITXBUF;
		DMA3SZ = len;
		DMA3CTL |= DMAEN__ENABLE;
		// Read RXBUF -> dst
		DMACTL2 |= DMA4TSEL__SPIRXIFG;
		DMA4CTL = DMADT_0 + DMADSTINCR_3 + DMASRCINCR_0 +  DMADSTBYTE__BYTE  + DMASRCBYTE__BYTE + DMALEVEL__EDGE;
		DMA4SA = &SPIRXBUF;
		__data16_write_addr(&DMA4DA, dst); /* direct assignment does not work for 20-bit addresses with GCC */
		DMA4SZ = len;
		DMA4CTL |= DMAEN__ENABLE;
		//Trigger TX DMA
		SPIIFG &= ~UCTXIFG;
		SPIIFG |=  UCTXIFG;
//		while(SPISTATW & 0x1);
		//Trigger RX DMA
//		SPIIFG &= ~UCRXIFG;
//		SPIIFG |=  UCRXIFG;
		while(DMA4CTL & DMAEN__ENABLE);
		while(DMA3CTL & DMAEN__ENABLE);

#else // !defined(__MSP430__)
		MAP_DMA_enableModule();
		MAP_DMA_setControlBase(controlTable);

		MAP_DMA_assignChannel(MSP432_DMA_EUSCI_TRANSMIT_CHANNEL);
		MAP_DMA_assignChannel(MSP432_DMA_EUSCI_RECEIVE_CHANNEL);

		MAP_DMA_disableChannelAttribute(MSP432_DMA_EUSCI_TRANSMIT_CHANNEL,
			UDMA_ATTR_ALTSELECT | UDMA_ATTR_USEBURST | UDMA_ATTR_HIGH_PRIORITY | UDMA_ATTR_REQMASK);
		MAP_DMA_disableChannelAttribute(MSP432_DMA_EUSCI_RECEIVE_CHANNEL,
			UDMA_ATTR_ALTSELECT | UDMA_ATTR_USEBURST | UDMA_ATTR_HIGH_PRIORITY | UDMA_ATTR_REQMASK);

		MAP_DMA_enableChannelAttribute(MSP432_DMA_EUSCI_TRANSMIT_CHANNEL, UDMA_ATTR_HIGH_PRIORITY);

		MAP_DMA_setChannelControl(
			MSP432_DMA_EUSCI_TRANSMIT_CHANNEL | UDMA_PRI_SELECT,
			UDMA_ARB_1 | UDMA_SIZE_8 | UDMA_SRC_INC_NONE | UDMA_DST_INC_NONE
		);
		MAP_DMA_setChannelControl(
			MSP432_DMA_EUSCI_RECEIVE_CHANNEL | UDMA_PRI_SELECT,
			UDMA_ARB_1 | UDMA_SIZE_8 | UDMA_SRC_INC_NONE | UDMA_DST_INC_8
		);

		// MAP_DMA_assignInterrupt(DMA_INT1, MSP432_DMA_EUSCI_TRANSMIT_CHANNEL_NUM);
		MAP_DMA_assignInterrupt(DMA_INT2, MSP432_DMA_EUSCI_RECEIVE_CHANNEL_NUM);

		// MAP_Interrupt_enableInterrupt(INT_DMA_INT1);
		MAP_Interrupt_enableInterrupt(INT_DMA_INT2);

		MAP_Interrupt_disableSleepOnIsrExit(); // XXX keep this?

		MAP_DMA_setChannelTransfer(
			MSP432_DMA_EUSCI_TRANSMIT_CHANNEL | UDMA_PRI_SELECT, UDMA_MODE_BASIC,
			&dummy,
			(void*) MAP_SPI_getTransmitBufferAddressForDMA(MSP432_DMA_EUSCI_MODULE),
			len
		);
		MAP_DMA_setChannelTransfer(
			MSP432_DMA_EUSCI_RECEIVE_CHANNEL | UDMA_PRI_SELECT, UDMA_MODE_BASIC,
			(void*) MAP_SPI_getReceiveBufferAddressForDMA(MSP432_DMA_EUSCI_MODULE), (void*) dst, len
		);

		curDMATransmitChannelNum = MSP432_DMA_EUSCI_TRANSMIT_CHANNEL_NUM;
		curDMAReceiveChannelNum = MSP432_DMA_EUSCI_RECEIVE_CHANNEL_NUM;

		// Write to DMA register directly instead of using driverlib functions
		// to make sure both transmit and receive DMA channels are enabled *simultaneously*
		SPIIFG &= ~UCRXIFG;
		// DMA_Control->ENASET |= ((1 << MSP432_DMA_EUSCI_TRANSMIT_CHANNEL_NUM) | (1 << MSP432_DMA_EUSCI_RECEIVE_CHANNEL_NUM));
		MAP_DMA_enableChannel(MSP432_DMA_EUSCI_TRANSMIT_CHANNEL_NUM);
		MAP_DMA_enableChannel(MSP432_DMA_EUSCI_RECEIVE_CHANNEL_NUM);

		while (MAP_DMA_isChannelEnabled(MSP432_DMA_EUSCI_TRANSMIT_CHANNEL_NUM)) {}
		while (MAP_DMA_isChannelEnabled(MSP432_DMA_EUSCI_RECEIVE_CHANNEL_NUM)) {}
#endif
	while(SPISTATW & 0x1);
	SLAVE_CS_OUT |= SLAVE_CS_PIN;
#elif defined(__STM32__)
	FRAM_Read(A->L, dst, len);
#endif
}

void SPI_WAIT_DMA(void) {
#ifdef __EXT_FRAM_MSP__
#ifdef __MSP430__
	while (DMA3CTL & DMAEN__ENABLE);
#endif
#ifdef __MSP432__
	while (MAP_DMA_isChannelEnabled(curDMATransmitChannelNum)) {}
#endif
	if (dma_timer_delay) {
		TA1CTL = TIMER_A_STOP_MODE + TIMER_A_DO_CLEAR;
		dma_timer_delay = 0;
	}
	// wait for the last byte to be written
	while(SPISTATW & 0x1);
	SLAVE_CS_OUT |= SLAVE_CS_PIN;
#endif
}

void SPI_WRITE(SPI_ADDR* A, const uint8_t *src, unsigned long len ){
	SPI_WRITE2(A, src, len, 0);
	SPI_WAIT_DMA();
}

void SPI_WRITE2(SPI_ADDR* A, const uint8_t *src, unsigned long len, uint16_t timer_delay) {
#ifdef __EXT_FRAM_MSP__
	//All writes to the memory begin with a WREN opcode with CS being asserted and deasserted.
	SLAVE_CS_OUT &= ~(SLAVE_CS_PIN);
		SPITXBUF = CMD_WREN;
		while(SPISTATW & 0x1); //SPI BUSY
	SLAVE_CS_OUT |= SLAVE_CS_PIN;

	SLAVE_CS_OUT &= ~(SLAVE_CS_PIN);
		SPITXBUF = CMD_WRITE;
		while(SPISTATW & 0x1);
		SPITXBUF=A->byte[2];
		while(SPISTATW & 0x1);
		SPITXBUF=A->byte[1];
		while(SPISTATW & 0x1);
		SPITXBUF=A->byte[0];
		while(SPISTATW & 0x1);

#ifdef __MSP430__

		if (!timer_delay) {
			//Triggered when TX is done
			DMACTL1 = (DMACTL1 & 0x00ff) | DMA3TSEL__SPITXIFG;
		} else {
			DMACTL1 = (DMACTL1 & 0x00ff) | DMA3TSEL__TA1CCR0;
		}
		dma_timer_delay = timer_delay;
		DMA3CTL = DMADT_0 + DMADSTINCR_0 + DMASRCINCR_3 +  DMADSTBYTE__BYTE  + DMASRCBYTE__BYTE + DMALEVEL__EDGE + DMAIE;
		__data16_write_addr(&DMA3SA, src); /* direct assignment does not work for 20-bit addresses with GCC */
		DMA3DA = &SPITXBUF;
		DMA3SZ = len;
		DMA3CTL |= DMAEN__ENABLE;
		if (!timer_delay) {
			//Triger the DMA to invoke the first transfer
			SPIIFG &= ~UCTXIFG;
			SPIIFG |=  UCTXIFG;
		}
		if (timer_delay) {
			// Timer A0 is already used by FreeRTOS
			TA1CCTL0 = TIMER_A_OUTPUTMODE_TOGGLE;
			TA1CCR0 = timer_delay;
			TA1CCR1 = 1; // a random number smaller than TA1CCR0; for generating PWM signals
			TA1CTL = TIMER_A_CLOCKSOURCE_SMCLK + TIMER_A_CLOCKSOURCE_DIVIDER_1 + TIMER_A_UP_MODE;
		}
		__bis_SR_register(GIE);
#else
		// Ref: dma_eusci_spi.c from https://e2e.ti.com/support/microcontrollers/msp430/f/166/t/453110?MSP432-SPI-with-DMA
		MAP_DMA_enableModule();
		MAP_DMA_setControlBase(controlTable);
		dma_timer_delay = timer_delay;
		uint32_t dma_mapping, dma_channel;
		if (!timer_delay) {
			dma_mapping = MSP432_DMA_EUSCI_TRANSMIT_CHANNEL;
			dma_channel = MSP432_DMA_EUSCI_TRANSMIT_CHANNEL_NUM;
		} else {
			dma_mapping = DMA_CH3_TIMERA1CCR2;
			dma_channel = 3;
		}
		MAP_DMA_assignChannel(dma_mapping);
		MAP_DMA_disableChannelAttribute(dma_mapping,
			UDMA_ATTR_ALTSELECT | UDMA_ATTR_USEBURST | UDMA_ATTR_HIGH_PRIORITY | UDMA_ATTR_REQMASK);
		MAP_DMA_setChannelControl(
			dma_mapping | UDMA_PRI_SELECT,
			UDMA_ARB_1 | UDMA_SIZE_8 | UDMA_SRC_INC_8 | UDMA_DST_INC_NONE
		);
		MAP_DMA_assignInterrupt(DMA_INT1, dma_channel);
		MAP_Interrupt_enableInterrupt(INT_DMA_INT1);
		MAP_Interrupt_disableSleepOnIsrExit(); // XXX keep this?
		MAP_DMA_setChannelTransfer(
			dma_mapping | UDMA_PRI_SELECT, UDMA_MODE_BASIC,
			(void*) src, (void*) MAP_SPI_getTransmitBufferAddressForDMA(MSP432_DMA_EUSCI_MODULE), len
		);
		curDMATransmitChannelNum = dma_channel;
		curDMAReceiveChannelNum = 0;
		MAP_DMA_enableChannel(dma_channel);
		if (timer_delay) {
			TA1CCTL0 = TIMER_A_OUTPUTMODE_TOGGLE;
			TA1CCR0 = timer_delay;
			TA1CCR1 = 1;
			TA1CTL = TIMER_A_CLOCKSOURCE_SMCLK + TIMER_A_CLOCKSOURCE_DIVIDER_1 + TIMER_A_UP_MODE;
		}
#endif
#elif defined(__STM32__)
	FRAM_Write(A->L, src, len);
#endif
}

#ifdef __MSP430__

#pragma vector=DMA_VECTOR
__interrupt void DMA_ISR(void)
{
	DMA3CTL &= ~DMAIE;
}

#endif

#ifdef __MSP432__

void DMA_INT1_IRQHandler(void) {
    MAP_DMA_clearInterruptFlag(curDMATransmitChannelNum);
    MAP_DMA_disableInterrupt(DMA_INT1);
    MAP_Interrupt_disableInterrupt(DMA_INT1);
    MAP_DMA_disableChannel(curDMATransmitChannelNum);
}

void DMA_INT2_IRQHandler(void) {
    MAP_DMA_clearInterruptFlag(curDMAReceiveChannelNum);
    MAP_DMA_clearInterruptFlag(curDMATransmitChannelNum);
    MAP_DMA_disableInterrupt(DMA_INT2);
    MAP_Interrupt_disableInterrupt(DMA_INT2);
    MAP_DMA_disableChannel(curDMAReceiveChannelNum);
    MAP_DMA_disableChannel(curDMATransmitChannelNum);
}

#endif

#define TEST_ARRAY_LEN 16

uint8_t testSPI(void) {
    SPI_ADDR A;
    uint8_t test_array[TEST_ARRAY_LEN];
    A.L = 0;
    SPI_READ( &A, test_array, TEST_ARRAY_LEN );
    for (uint8_t idx = 0; idx < TEST_ARRAY_LEN; idx++) {
        test_array[idx] = idx;
    }
    SPI_WRITE( &A, test_array, TEST_ARRAY_LEN );
    memset(test_array, 0xFF, TEST_ARRAY_LEN);
    SPI_READ( &A, test_array, TEST_ARRAY_LEN );
    for (uint8_t idx = 0; idx < TEST_ARRAY_LEN; idx++) {
        if (test_array[idx] != idx) {
            return 1;
        }
    }
    return 0;
}
