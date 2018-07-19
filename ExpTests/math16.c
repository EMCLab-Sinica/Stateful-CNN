/*
 * math16.c
 *
 *  Created on: 2018¦~3¤ë19¤é
 *      Author: Meenchen
 */

#include <config.h>
extern unsigned long timeCounter;
extern unsigned long taskRecency[12];
extern unsigned long information[10];

/*******************************************************************************
*
* Name : 16-bit Math
* Purpose : Benchmark 16-bit math functions.
*
*******************************************************************************/
typedef unsigned short UInt16;
UInt16 add16(UInt16 a, UInt16 b)
{
    return (a + b);
}
UInt16 mul16(UInt16 a, UInt16 b)
{
    return (a * b);
}
UInt16 div16(UInt16 a, UInt16 b)
{
    return (a / b);
}

#pragma NOINIT(Fresult)
volatile UInt16 Fresult[4];
volatile UInt16 Sresult[4];


#ifdef ONVM
void readback_m16()
{
    int j;
    for(j = 0; j < 4; j++){
        Sresult[j] = Fresult[j];
    }
}
#endif



void math16()
{
    int k;

#ifdef ONVM
    readback_m16();
#endif

    while(1){
        for(k = 0; k < ITERMATH16; k++){
#ifdef ONNVM
            Fresult[0] = 231;
            Fresult[1] = 12;
            Fresult[2] = Fresult[0] + Fresult[1];
            Fresult[1] = Fresult[0] * Fresult[2];
            Fresult[3] = Fresult[1] / Fresult[2];
#endif
#ifdef OUR
            Sresult[0] = 231;
            Sresult[1] = 12;
            Sresult[2] = Sresult[0] + Sresult[1];
            Sresult[1] = Sresult[0] * Sresult[2];
            Fresult[3] = Sresult[1] / Sresult[2];
#endif
#ifdef ONEVERSION
            Sresult[0] = 231;
            Sresult[1] = 12;
            Sresult[2] = Fresult[0] + Fresult[1];
            Sresult[1] = Fresult[0] * Fresult[2];
            Fresult[3] = Fresult[1] / Fresult[2];
#endif
#ifdef ONVM
            Sresult[0] = 231;
            Sresult[1] = 12;
            Sresult[2] = Sresult[0] + Sresult[1];
            Sresult[1] = Sresult[0] * Sresult[2];
            Fresult[3] = Sresult[1] / Sresult[2];
#endif
        }
        information[IDMATH16]++;
    }
}

