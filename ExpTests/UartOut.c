/*
 * UartOut.c
 *
 *  Created on: 2018¦~3¤ë13¤é
 *      Author: Meenchen
 */
#include <ExpTests/UartOut.h>
#include <Tools/myuart.h>
#include <FreeRTOS.h>
#include <task.h>
#include <config.h>

extern int avgtempID;
extern int avgcapID;

extern unsigned long information[10];

extern unsigned long timeCounter;
unsigned long uartCounter;

void DumpOut()
{
    while(avgcapID == -1 || avgtempID == -1);//not ready for both sensor
    while(1){
        //Output sensor values
        int cap,temp;
        DBreadIn(&cap, avgcapID);
        DBreadIn(&temp, avgtempID);
//        print2uart("Vol: %d, Temp: %d\r\n",cap,temp);

        //Output tasks' information
        print2uart(":%l,%l,%l,%l,%l,%l,%l,%l,%l,%l\r\n", information[0],information[1],information[2],information[3],information[4],information[5],information[6],information[7],information[8],information[9]);
        dprint2uart("xPortGetFreeHeapSize = %l\r\n", xPortGetFreeHeapSize());
        information[IDUART]++;
        //Wait for next x Ticks
        while(uartCounter+OUT_TICKS > timeCounter){
            portYIELD();
        }
        uartCounter = timeCounter;
    }
}
