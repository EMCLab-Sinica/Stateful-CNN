#ifndef JAPARILIB_EXTFRAM_H_
#define JAPARILIB_EXTFRAM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
	
#define EXTFRAM_USE_DMA

extern uint8_t controlTable[1024] __attribute__((aligned (1024)));
extern uint32_t curDMATransmitChannelNum, curDMAReceiveChannelNum;

typedef union Data {
    unsigned char byte[4];
    unsigned long L;
}SPI_ADDR;

void eraseFRAM(void);
void eraseFRAM2(uint8_t init_val);
void initSPI(void);
void SPI_READ(SPI_ADDR* A,uint8_t *dst, unsigned long len );
void SPI_WAIT_DMA();
void SPI_WRITE(SPI_ADDR* A, const uint8_t *src, unsigned long len );
void SPI_WRITE2(SPI_ADDR* A, const uint8_t *src, unsigned long len, uint16_t timer_delay );
uint8_t testSPI(void);

#ifdef __cplusplus
}
#endif

#endif /* JAPARILIB_EXTFRAM_H_ */
