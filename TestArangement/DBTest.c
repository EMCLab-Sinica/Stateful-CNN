/*
 * DBTest.c
 *
 *  Created on: 2017¦~9¤ë1¤é
 *      Author: Meenchen
 */

#include <DataManager/SimpDB.h>
#include <FreeRTOS.h>
#include <task.h>
#include "../ExpTests/Tester.h"

#define COUNTER_TASK_PRIORITY     ( tskIDLE_PRIORITY )

/* main function to create testing unit, just keep  */
void main_DBtest( void );

/* Testing uint  */
static void prvTest( void *pvParameters );
static void prvTest1( void *pvParameters );
static void prvTest2( void *pvParameters );

extern int firstTime;
int id;
/*-----------------------------------------------------------*/

void main_DBtest( void )
{
    /* Start a simple task for test */
    if(firstTime != 1){
        constructor();
        DBmodeSelect(VM);
    }

    xTaskCreate( calibrateCap, "calibrateCap", configMINIMAL_STACK_SIZE, NULL, COUNTER_TASK_PRIORITY, NULL );
    xTaskCreate( calibrateTemp, "calibrateTemp", configMINIMAL_STACK_SIZE, NULL, COUNTER_TASK_PRIORITY, NULL );
    xTaskCreate( SenseLog, "SenseLog", configMINIMAL_STACK_SIZE, NULL, COUNTER_TASK_PRIORITY, NULL );
    xTaskCreate( CapLog, "CapLog", configMINIMAL_STACK_SIZE, NULL, COUNTER_TASK_PRIORITY, NULL );
    xTaskCreate( DumpOut, "DumpOut", configMINIMAL_STACK_SIZE, NULL, COUNTER_TASK_PRIORITY, NULL );
    xTaskCreate( firFilter, "firFilter", configMINIMAL_STACK_SIZE, NULL, COUNTER_TASK_PRIORITY, NULL );
    xTaskCreate( math16, "math16", configMINIMAL_STACK_SIZE, NULL, COUNTER_TASK_PRIORITY, NULL );
    xTaskCreate( math32, "math32", configMINIMAL_STACK_SIZE, NULL, COUNTER_TASK_PRIORITY, NULL );
    xTaskCreate( dimMatrix, "dimMatrix", configMINIMAL_STACK_SIZE, NULL, COUNTER_TASK_PRIORITY, NULL );
    xTaskCreate( matrixmultiplication, "matrixmultiplication", configMINIMAL_STACK_SIZE, NULL, COUNTER_TASK_PRIORITY, NULL );

    /* Start the scheduler. */
    vTaskStartScheduler();

    /* If all is well, the scheduler will now be running, and the following
    line will never be reached.  If the following line does execute, then
    there was insufficient FreeRTOS heap memory available for the Idle and/or
    timer tasks to be created.  See the memory management section on the
    FreeRTOS web site for more details on the FreeRTOS heap
    http://www.freertos.org/a00111.html. */
    for( ;; );
}

static void prvTest( void *pvParameters )
{
    int i;
    struct working dataA;

    constructor();
//    registerTCB();
    DBmodeSelect(VM);
    DBworking(&dataA, 20, -1);//20 bytes and for create(-1)
    unsigned char* pointA = (unsigned char*)dataA.address;
    for(i = 0; i < 20; i++){
        pointA[i] = i;
    }
    dataA.id = DBcommit(&dataA,NULL,NULL,20,1);
    id = dataA.id;
//    unresgisterTCB();

    //Create the second task for testing concurrency control
    xTaskCreate( prvTest1, "DBTest2", configMINIMAL_STACK_SIZE, NULL, COUNTER_TASK_PRIORITY, NULL );
    xTaskCreate( prvTest2, "DBTest2", configMINIMAL_STACK_SIZE, NULL, COUNTER_TASK_PRIORITY, NULL );

    vTaskDelete(NULL);//delete the current TCB
}

static void prvTest1( void *pvParameters )
{
    int i;
    struct working dataA;

    unsigned char* pointA;

    while(1){
//        registerTCB();
        DBworking(&dataA, 20, 0);
        DBreadIn(dataA.address, id);
        pointA = (unsigned char*)dataA.address;
        for(i = 0; i < 20; i++){
            pointA[i] *= 2;
        }
        DBcommit(&dataA,NULL,NULL,20,1);
//        unresgisterTCB();
    }
}

static void prvTest2( void *pvParameters )
{
    int i;
    struct working dataA;
    dataA.id = 0;

    while(1){
//        registerTCB();
        DBworking(&dataA, 20, 0);
        DBreadIn(dataA.address, id);
        unsigned char* pointA = (unsigned char*)dataA.address;
        pointA = (unsigned char*)dataA.address;
        for(i = 0; i < 20; i++){
            pointA[i]++;
        }
        DBcommit(&dataA,NULL,NULL,20,1);
//        unresgisterTCB();
    }
}
