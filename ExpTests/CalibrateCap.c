/*
 * CalibrateCap.c
 *
 *  Created on: 2018/3/15
 *      Author: Meenchen
 */
#include <FreeRTOS.h>
#include <task.h>
#include <DataManager/SimpDB.h>
#include <config.h>

extern int capID;
extern int avgcapID;
extern unsigned long information[10];
extern int readCap;
int averageCap = 0;

void calibrateCap(){
    //Wait for the first data
    while(capID < 0)portYIELD();

    while(1){
        while(readCap <= 0)portYIELD();
        readCap = 0;
        registerTCB(IDCAPCALIBRATE);

        //read most recent temperature
        int t;
        DBreadIn(&t, capID);
        if(averageCap == 0)
            averageCap = t;
        else
            averageCap = (averageCap + t)/2;

        //commit the average value
        struct working data;
        DBworking(&data, 2, avgcapID);
        int* ptr = data.address;
        *ptr = averageCap;
        avgcapID = DBcommit(&data,NULL,NULL,2,1);

        information[IDCAPCALIBRATE]++;
        unresgisterTCB(IDCAPCALIBRATE);
        portYIELD();
    }
}
