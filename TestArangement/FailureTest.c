/*
 * FailureTest.c
 *
 *  Created on: 2017¦~9¤ë1¤é
 *      Author: Meenchen
 */


#include <DataManager/SimpDB.h>
#include "FreeRTOS.h"
#include "task.h"

#define COUNTER_TASK_PRIORITY     ( tskIDLE_PRIORITY + 1 )

/* main function to create testing unit, just keep  */
void main_failureTest( void );

/* Testing uint  */
static void prvTest( void *pvParameters );

#pragma DATA_SECTION(firstTime, ".map") //indicate whether task stacks exist
static int firstTime;

extern int information[100];

/*-----------------------------------------------------------*/

void main_failureTest( void )
{
    /* Start a simple task for test */
    xTaskCreate( prvTest, "FailureTest", configMINIMAL_STACK_SIZE, NULL, COUNTER_TASK_PRIORITY, NULL );

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
    int i,j;
    volatile int counter,ouputer;

    ouputer = information[0];
//    constructor();
//    DBworking(&dataA, 20, -1);//20 bytes and for create(-1)
//    unsigned char* pointA = (unsigned char*)dataA.address;
//    for(i = 0; i < 20; i++){
//        pointA[i] = i;
//    }
//    DBcommit(&dataA,NULL,NULL,20,1);
    while(1){
        for(i = 0; i < 1000;i++){
            for(j = 0; j < 1000;j++){
                counter++;
            }
        }
        ouputer++;
        information[0] = ouputer;
    }
}
