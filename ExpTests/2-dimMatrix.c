/*
 * 2-dimMatrix.c
 *
 *  Created on: 2018¦~3¤ë19¤é
 *      Author: Meenchen
 */

#include <config.h>
#include <DataManager/SimpDB.h>
extern unsigned long timeCounter;
extern unsigned long taskRecency[12];
extern unsigned long information[10];

/*******************************************************************************
*
* Name : 16-bit 2-dim Matrix
* Purpose : Benchmark copying 16-bit values.
*
*******************************************************************************/
typedef unsigned short UInt16;
#pragma NOINIT(Fmm)
volatile UInt16 Fmm[16][4] = {{0x1234, 0x5678, 0x9012, 0x3456},
                             {0x7890, 0x1234, 0x5678, 0x9012},
                             {0x3456, 0x7890, 0x1234, 0x5678},
                             {0x9012, 0x3456, 0x7890, 0x1234},
                             {0x1234, 0x5678, 0x9012, 0x3456},
                             {0x7890, 0x1234, 0x5678, 0x9012},
                             {0x3456, 0x7890, 0x1234, 0x5678},
                             {0x9012, 0x3456, 0x7890, 0x1234},
                             {0x1234, 0x5678, 0x9012, 0x3456},
                             {0x7890, 0x1234, 0x5678, 0x9012},
                             {0x3456, 0x7890, 0x1234, 0x5678},
                             {0x9012, 0x3456, 0x7890, 0x1234},
                             {0x1234, 0x5678, 0x9012, 0x3456},
                             {0x7890, 0x1234, 0x5678, 0x9012},
                             {0x3456, 0x7890, 0x1234, 0x5678},
                             {0x9012, 0x3456, 0x7890, 0x1234}
                             };;
#pragma NOINIT(Fmm2)
volatile UInt16 Fmm2[16][4];
#pragma NOINIT(Fmm3)
volatile UInt16 Fmm3[16][4];
#pragma NOINIT(Fmm4)
volatile UInt16 Fmm4[16][4];

volatile UInt16 Smm[16][4];
volatile UInt16 Smm2[16][4];
volatile UInt16 Smm3[16][4];

void dimMatrix()
{
    int i, j, k;

    while(1){
        for(k = 0; k < ITER2DMATRIX; k++){
            for(i = 0; i < 16; i++)
            {
                for(j = 0; j < 4; j++)
                {
                    Smm2[i][j] = Fmm[i][j];
                    Smm3[i][j] = Smm2[i][j];
                    Smm2[i][j] = Smm3[i][j];
                    Fmm[i][j] = Smm2[i][j];
                }
            }
        }
        information[ID2DMATRIX]++;
    }
}


