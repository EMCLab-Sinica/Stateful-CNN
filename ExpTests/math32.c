/*
 * math32.h
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
* Name : 32-bit Math
* Purpose : Benchmark 32-bit math functions.
*
*******************************************************************************/
#include <math.h>
typedef unsigned long UInt32;
UInt32 add(UInt32 a, UInt32 b)
{
    return (a + b);
}
UInt32 mul(UInt32 a, UInt32 b)
{
    return (a * b);
}
UInt32 div(UInt32 a, UInt32 b)
{
    return (a / b);
}

#pragma NOINIT(Fresult32)
volatile UInt32 Fresult32[4];
volatile UInt32 Sresult32[4];

#ifdef ONVM
void readback_m32()
{
    int j;
    for(j = 0; j < 4; j++){
        Sresult32[j] = Fresult32[j];
    }
}
#endif


void math32()
{
    int i;

#ifdef ONVM
    readback_m32();
#endif

    while(1){
        for(i = 0; i < ITERMATH32; i++){
#ifdef ONNVM
            Fresult32[0] = 43125;
            Fresult32[1] = 14567;
            Fresult32[2] = Fresult32[0] + Fresult32[1];
            Fresult32[1] = Fresult32[0] * Fresult32[2];
            Fresult32[3] = Fresult32[1] / Fresult32[2];
#endif
#ifdef OUR
            Sresult32[0] = 43125;
            Sresult32[1] = 14567;
            Sresult32[2] = Sresult32[0] + Sresult32[1];
            Sresult32[1] = Sresult32[0] * Sresult32[2];
            Fresult32[3] = Sresult32[1] / Sresult32[2];
#endif
#ifdef ONEVERSION
            Sresult32[0] = 43125;
            Sresult32[1] = 14567;
            Sresult32[2] = Sresult32[0] + Sresult32[1];
            Sresult32[1] = Sresult32[0] * Sresult32[2];
            Fresult32[3] = Fresult32[1] / Sresult32[2];
#endif
#ifdef ONVM
            Sresult32[0] = 43125;
            Sresult32[1] = 14567;
            Sresult32[2] = Sresult32[0] + Sresult32[1];
            Sresult32[1] = Sresult32[0] * Sresult32[2];
            Fresult32[3] = Sresult32[1] / Sresult32[2];
#endif
        }
        information[IDMATH32]++;
    }
}

