#include <DataManager/SimpDB.h>
#include "StackInVM.h"
#include "FreeRTOS.h"
#include "task.h"

#define COUNTER_TASK_PRIORITY     ( tskIDLE_PRIORITY + 1 )

/* main function to create testing unit, just keep  */
void main_counting( void );

/* Testing uint  */
static void prvCount( void *pvParameters );

/*-----------------------------------------------------------*/

void main_counting( void )
{
    /* Start a simple task for test */
    xTaskCreate( prvCount, "Count1", configMINIMAL_STACK_SIZE, NULL, COUNTER_TASK_PRIORITY, NULL );

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

static void prvCount( void *pvParameters )
{
    long a[10];
    int i;
    for(i = 0; i < 10; i++) a[i] = 0;
    for( ; ; ){
        a[0]++;
        a[1]++;
        a[2]++;
        a[3]++;
        a[4]++;
        a[5]++;
        a[9]++;
        a[8]++;
        a[7]++;
        a[6]++;
        i++;
        if(i > 10000){
            information[0]++;
            i = 0;
        }
        if(a[9] >= 10000000){//10000000
            break;
        }
    }
    for(i = 0; i < 10; i++) a[i] = 0;
}
