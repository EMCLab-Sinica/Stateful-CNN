/*
 * CalibrateTemp.c
 *
 *  Created on: 2018/3/15
 *      Author: Meenchen
 */
#include <FreeRTOS.h>
#include <task.h>
#include <DataManager/SimpDB.h>
#include <config.h>

extern int avgtempID;
extern int tempID;
extern unsigned long information[10];
extern int readTemp;

int averageTemp = 0;

void calibrateTemp(){
    //Wait for the first data
    while(tempID < 0)portYIELD();

    while(1){
        while(readTemp <= 0)portYIELD();
        readTemp = 0;
        registerTCB(IDTEMPCALIBRATE);

        //read most recent temperature
        float t;
        DBreadIn(&t, tempID);
        if(averageTemp == 0)
            averageTemp = t;
        else
            averageTemp = (averageTemp + (int)t)/2;

        //commit the average value
        struct working data;
        DBworking(&data, 2, avgtempID);
        int* ptr = data.address;
        *ptr = averageTemp;
        avgtempID = DBcommit(&data,NULL,NULL,2,1);

        information[IDTEMPCALIBRATE]++;
        unresgisterTCB(IDTEMPCALIBRATE);
        portYIELD();
    }
}
