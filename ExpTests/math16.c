/*
 * math16.c
 *
 *  Created on: 2018/3/19
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


void math16()
{
    int k;

    while(1){
        for(k = 0; k < ITERMATH16; k++){
            Sresult[0] = 231;
            Sresult[1] = 12;
            Sresult[2] = Sresult[0] + Sresult[1];
            Sresult[1] = Sresult[0] * Sresult[2];
            Fresult[3] = Sresult[1] / Sresult[2];
        }
        information[IDMATH16]++;
    }
}

