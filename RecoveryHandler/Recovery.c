/*
 * Recovery.c
 *
 *  Created on: 2018/2/12
 *      Author: Meenchen
 */

#include <RecoveryHandler/Recovery.h>
#include "FreeRTOS.h"
#include <stdio.h>
#include "task.h"
#include "Tools/myuart.h"
#include "driverlib.h"

/* Used to check whether memory address is valid */
#define heapBITS_PER_BYTE       ( ( size_t ) 8 )
typedef struct A_BLOCK_LINK
{
    struct A_BLOCK_LINK *pxNextFreeBlock;   /*<< The next free block in the list. */
    size_t xBlockSize;                      /*<< The size of the free block. */
} BlockLink_t;
static const size_t xHeapStructSize = ( sizeof( BlockLink_t ) + ( ( size_t ) ( portBYTE_ALIGNMENT - 1 ) ) ) & ~( ( size_t ) portBYTE_ALIGNMENT_MASK );
const size_t xBlockAllocatedBit = ( ( size_t ) 1 ) << ( ( sizeof( size_t ) * heapBITS_PER_BYTE ) - 1 );

/*
 * Task control block.  A task control block (TCB) is allocated for each task,
 * and stores task state information, including a pointer to the task's context
 * (the task's run time environment, including register values)
 */
typedef struct tskTaskControlBlock
{
    volatile StackType_t    *pxTopOfStack;  /*< Points to the location of the last item placed on the tasks stack.  THIS MUST BE THE FIRST MEMBER OF THE TCB STRUCT. */

    #if ( portUSING_MPU_WRAPPERS == 1 )
        xMPU_SETTINGS   xMPUSettings;       /*< The MPU settings are defined as part of the port layer.  THIS MUST BE THE SECOND MEMBER OF THE TCB STRUCT. */
    #endif

    ListItem_t          xStateListItem; /*< The list that the state list item of a task is reference from denotes the state of that task (Ready, Blocked, Suspended ). */
    ListItem_t          xEventListItem;     /*< Used to reference a task from an event list. */
    UBaseType_t         uxPriority;         /*< The priority of the task.  0 is the lowest priority. */
    StackType_t         *pxStack;           /*< Points to the start of the stack. */
    char                pcTaskName[ configMAX_TASK_NAME_LEN ];/*< Descriptive name given to the task when created.  Facilitates debugging only. */ /*lint !e971 Unqualified char types are allowed for strings and single characters only. */

    /*------------------------------  Extend to support validation: Start ------------------------------*/
    unsigned long vBegin;
    unsigned long vEnd;
    /*------------------------------  Extend to support validation: End ------------------------------*/
    /*------------------------------  Extend to support dynamic stack: Start ------------------------------*/
    void * AddressOfVMStack;
    void * AddressOffset;
    int StackInNVM;
    int initial;
    /*------------------------------  Extend to support dynamic stack: End ------------------------------*/
    /*------------------------------  Extend to support dynamic function: Start ------------------------------*/
    void * AddressOfNVMFunction;
    void * AddressOfVMFunction;
    void * CodeOffset;
    int SizeOfFunction;
    int CodeInNVM;
    /*------------------------------  Extend to support dynamic function: End ------------------------------*/

    #if ( portSTACK_GROWTH > 0 )
        StackType_t     *pxEndOfStack;      /*< Points to the end of the stack on architectures where the stack grows up from low memory. */
    #endif

    #if ( portCRITICAL_NESTING_IN_TCB == 1 )
        UBaseType_t     uxCriticalNesting;  /*< Holds the critical section nesting depth for ports that do not maintain their own count in the port layer. */
    #endif

    #if ( configUSE_TRACE_FACILITY == 1 )
        UBaseType_t     uxTCBNumber;        /*< Stores a number that increments each time a TCB is created.  It allows debuggers to determine when a task has been deleted and then recreated. */
        UBaseType_t     uxTaskNumber;       /*< Stores a number specifically for use by third party trace code. */
    #endif

    #if ( configUSE_MUTEXES == 1 )
        UBaseType_t     uxBasePriority;     /*< The priority last assigned to the task - used by the priority inheritance mechanism. */
        UBaseType_t     uxMutexesHeld;
    #endif

    #if ( configUSE_APPLICATION_TASK_TAG == 1 )
        TaskHookFunction_t pxTaskTag;
    #endif

    #if( configNUM_THREAD_LOCAL_STORAGE_POINTERS > 0 )
        void *pvThreadLocalStoragePointers[ configNUM_THREAD_LOCAL_STORAGE_POINTERS ];
    #endif

    #if( configGENERATE_RUN_TIME_STATS == 1 )
        uint32_t        ulRunTimeCounter;   /*< Stores the amount of time the task has spent in the Running state. */
    #endif

    #if ( configUSE_NEWLIB_REENTRANT == 1 )
        /* Allocate a Newlib reent structure that is specific to this task.
        Note Newlib support has been included by popular demand, but is not
        used by the FreeRTOS maintainers themselves.  FreeRTOS is not
        responsible for resulting newlib operation.  User must be familiar with
        newlib and must provide system-wide implementations of the necessary
        stubs. Be warned that (at the time of writing) the current newlib design
        implements a system-wide malloc() that must be provided with locks. */
        struct  _reent xNewLib_reent;
    #endif

    #if( configUSE_TASK_NOTIFICATIONS == 1 )
        volatile uint32_t ulNotifiedValue;
        volatile uint8_t ucNotifyState;
    #endif

    /* See the comments above the definition of
    tskSTATIC_AND_DYNAMIC_ALLOCATION_POSSIBLE. */
    #if( tskSTATIC_AND_DYNAMIC_ALLOCATION_POSSIBLE != 0 )
        uint8_t ucStaticallyAllocated;      /*< Set to pdTRUE if the task is a statically allocated to ensure no attempt is made to free the memory. */
    #endif

    #if( INCLUDE_xTaskAbortDelay == 1 )
        uint8_t ucDelayAborted;
    #endif

} tskTCB;

extern tskTCB * volatile pxCurrentTCB;
extern unsigned char volatile stopTrack;

/*
 * taskRerun(): rerun the current task invoking this function
 * parameters: none
 * return: none
 * note: Memory allocated by the task code is not automatically freed, and should be freed before the task is deleted
 * */
void taskRerun(){
    xTaskCreate( pxCurrentTCB->AddressOfNVMFunction, pxCurrentTCB->pcTaskName, configMINIMAL_STACK_SIZE, NULL, pxCurrentTCB->uxPriority, NULL);
    vTaskDelete(NULL);//delete the current TCB
}

/*
 * regTaskEnd(): mark the assigned task as started
 * parameters: none
 * return: none
 * */
void regTaskStart(void* add, unsigned short pri, unsigned short TCB, void* TCBA, int stopTrack){
    int i;
    for(i = 0; i < NUMTASK; i++){
        //find a invalid
        if(unfinished[i] != 1){
            address[i] = add;
            priority[i] = pri;
            TCBNum[i] = TCB;
            TCBAdd[i] = TCBA;
            schedulerTask[i] = stopTrack;
            unfinished[i] = 1;//incase failure before this
            break;
        }
    }
}


/*
 * regTaskEnd(): mark the current as ended
 * parameters: none
 * return: none
 * */
void regTaskEnd(){
    int i;
    for(i = 0; i < NUMTASK; i++){
        //find the slot
        if(unfinished[i] == 1 && TCBNum[i] == pxCurrentTCB->uxTCBNumber){
            unfinished[i] = 0;
            return;
        }
    }
}

/*
 * prvcheckAdd(): check if the pointer is actually allocated and can be freed
 * parameters: none
 * return: 1 for yes
 * */
int prvcheckAdd(void * pv){
    tskTCB * pxTCB = pv;
    uint8_t *puc = ( uint8_t * ) pxTCB->pxStack;
    BlockLink_t *pxLink;

    /* This casting is to keep the compiler from issuing warnings. */
    pxLink = ( void * ) (puc - xHeapStructSize);

    /* Check the block is actually allocated. */
    volatile unsigned long allocbit = pxLink->xBlockSize & xBlockAllocatedBit;
    if(allocbit == 0)
        return 0;
    if(pxLink->pxNextFreeBlock != NULL)
        return 0;

    return 1;
}

/*
 * freePreviousTasks(): free all unfinished tasks stacks after power failure, this only is used by default approach
 * parameters: none
 * return: none
 * */
void freePreviousTasks(){
    int i;
    for(i = 0; i < NUMTASK; i++){
        //find all unfinished tasks
        if(unfinished[i] == 1){
            //see if the address is balid
            if(prvcheckAdd(TCBAdd[i]) == 1){
                dprint2uart("Delete: %d\r\n", TCBNum[i]);
                //Since all tasks information, e.g., list of ready queue, is saved in VM, we only needs to consider the stack and free the stack and TCB
                tskTCB* tcb = TCBAdd[i];
                vPortFree(tcb->pxStack);
                vPortFree(tcb);
                unfinished[i] = 1;
            }
        }
    }
}


/*
 * failureRecovery(): recover all unfinished tasks after power failure
 * parameters: none
 * return: none
 * */
void failureRecovery(){
    int i;
//    stopTrack = 1;
    for(i = 0; i < NUMTASK; i++){
        //find all unfinished tasks
        if(unfinished[i] == 1){
            //see if the address is balid
            if(prvcheckAdd(TCBAdd[i]) == 1){
                dprint2uart("Recovery: Delete: %d\r\n", TCBNum[i]);
                //Since all tasks information, e.g., list of ready queue, is saved in VM, we only needs to consider the stack and free the stack and TCB
                tskTCB* tcb = TCBAdd[i];
                vPortFree(tcb->pxStack);
                vPortFree(tcb);
            }
            unfinished[i] = 0;
            if(!schedulerTask[i]){
                xTaskCreate(address[i], "Recovery", configMINIMAL_STACK_SIZE, NULL, priority[i], NULL);
                dprint2uart("Recovery Create: %d\r\n", TCBNum[i]);
            }
            dprint2uart("Rend: xPortGetFreeHeapSize = %l\r\n", xPortGetFreeHeapSize());
        }
    }
    /* Start the scheduler. */
    vTaskStartScheduler();
}
