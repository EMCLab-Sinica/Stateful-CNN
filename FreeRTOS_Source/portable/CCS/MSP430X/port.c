/*
    FreeRTOS V9.0.0 - Copyright (C) 2016 Real Time Engineers Ltd.
    All rights reserved

    VISIT http://www.FreeRTOS.org TO ENSURE YOU ARE USING THE LATEST VERSION.

    This file is part of the FreeRTOS distribution.

    FreeRTOS is free software; you can redistribute it and/or modify it under
    the terms of the GNU General Public License (version 2) as published by the
    Free Software Foundation >>>> AND MODIFIED BY <<<< the FreeRTOS exception.

    ***************************************************************************
    >>!   NOTE: The modification to the GPL is included to allow you to     !<<
    >>!   distribute a combined work that includes FreeRTOS without being   !<<
    >>!   obliged to provide the source code for proprietary components     !<<
    >>!   outside of the FreeRTOS kernel.                                   !<<
    ***************************************************************************

    FreeRTOS is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.  Full license text is available on the following
    link: http://www.freertos.org/a00114.html

    ***************************************************************************
     *                                                                       *
     *    FreeRTOS provides completely free yet professionally developed,    *
     *    robust, strictly quality controlled, supported, and cross          *
     *    platform software that is more than just the market leader, it     *
     *    is the industry's de facto standard.                               *
     *                                                                       *
     *    Help yourself get started quickly while simultaneously helping     *
     *    to support the FreeRTOS project by purchasing a FreeRTOS           *
     *    tutorial book, reference manual, or both:                          *
     *    http://www.FreeRTOS.org/Documentation                              *
     *                                                                       *
    ***************************************************************************

    http://www.FreeRTOS.org/FAQHelp.html - Having a problem?  Start by reading
    the FAQ page "My application does not run, what could be wrong?".  Have you
    defined configASSERT()?

    http://www.FreeRTOS.org/support - In return for receiving this top quality
    embedded software for free we request you assist our global community by
    participating in the support forum.

    http://www.FreeRTOS.org/training - Investing in training allows your team to
    be as productive as possible as early as possible.  Now you can receive
    FreeRTOS training directly from Richard Barry, CEO of Real Time Engineers
    Ltd, and the world's leading authority on the world's leading RTOS.

    http://www.FreeRTOS.org/plus - A selection of FreeRTOS ecosystem products,
    including FreeRTOS+Trace - an indispensable productivity tool, a DOS
    compatible FAT file system, and our tiny thread aware UDP/IP stack.

    http://www.FreeRTOS.org/labs - Where new FreeRTOS products go to incubate.
    Come and try FreeRTOS+TCP, our new open source TCP/IP stack for FreeRTOS.

    http://www.OpenRTOS.com - Real Time Engineers ltd. license FreeRTOS to High
    Integrity Systems ltd. to sell under the OpenRTOS brand.  Low cost OpenRTOS
    licenses offer ticketed support, indemnification and commercial middleware.

    http://www.SafeRTOS.com - High Integrity Systems also provide a safety
    engineered and independently SIL3 certified version for use in safety and
    mission critical applications that require provable dependability.

    1 tab == 4 spaces!
*/

/* Scheduler includes. */
#include "FreeRTOS.h"
#include "task.h"



//#define CHECKPOINT
#ifdef CHECKPOINT
#define PER 8//8 //every PERIOD =  PER * 2.5ms
extern unsigned char backup_sram[4096];
extern int exectued;
#endif

/*-----------------------------------------------------------
 * Implementation of functions defined in portable.h for the MSP430X port.
 *----------------------------------------------------------*/

/* Constants required for hardware setup.  The tick ISR runs off the ACLK,
not the MCLK. */
#define portACLK_FREQUENCY_HZ			( ( TickType_t ) 32768 )
#define portINITIAL_CRITICAL_NESTING	( ( uint16_t ) 10 )
#define portFLAGS_INT_ENABLED			( ( StackType_t ) 0x08 )

/* We require the address of the pxCurrentTCB variable, but don't want to know
any details of its type. */
//typedef void TCB_t;
//extern volatile TCB_t * volatile pxCurrentTCB;
//#define Meenchen
//#define CacheCode



#ifdef Meenchen
/*------------------------------  Extend to support dynamic stack: Start ------------------------------*/
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

void * pxAddressOfVMStack;
void * pxAddressOfNVMStack;
void * pxAddressOffset;
void * AddressSP;
void *pxSR,*pxPC,*temp;
extern int volatile StackToNVM;
extern int volatile CodeToNVM;
/*------------------------------  Extend to support dynamic stack: End ------------------------------*/
#endif

extern unsigned long timeCounter;
extern unsigned long taskRecency[12];
/* Each task maintains a count of the critical section nesting depth.  Each
time a critical section is entered the count is incremented.  Each time a
critical section is exited the count is decremented - with interrupts only
being re-enabled if the count is zero.

usCriticalNesting will get set to zero when the scheduler starts, but must
not be initialised to zero as this will cause problems during the startup
sequence. */
volatile uint16_t usCriticalNesting = portINITIAL_CRITICAL_NESTING;
/*-----------------------------------------------------------*/


/*
 * Sets up the periodic ISR used for the RTOS tick.  This uses timer 0, but
 * could have alternatively used the watchdog timer or timer 1.
 */
void vPortSetupTimerInterrupt( void );
/*-----------------------------------------------------------*/

/*
 * Initialise the stack of a task to look exactly as if a call to
 * portSAVE_CONTEXT had been called.
 *
 * See the header file portable.h.
 */
StackType_t *pxPortInitialiseStack( StackType_t *pxTopOfStack, TaskFunction_t pxCode, void *pvParameters )
{
uint16_t *pusTopOfStack;
uint32_t *pulTopOfStack, ulTemp;

	/*
		Place a few bytes of known values on the bottom of the stack.
		This is just useful for debugging and can be included if required.

		*pxTopOfStack = ( StackType_t ) 0x1111;
		pxTopOfStack--;
		*pxTopOfStack = ( StackType_t ) 0x2222;
		pxTopOfStack--;
		*pxTopOfStack = ( StackType_t ) 0x3333;
		pxTopOfStack--;
	*/

	/* Data types are need either 16 bits or 32 bits depending on the data 
	and code model used. */
	if( sizeof( pxCode ) == sizeof( uint16_t ) )
	{
		pusTopOfStack = ( uint16_t * ) pxTopOfStack;
		ulTemp = ( uint32_t ) pxCode;
		*pusTopOfStack = ( uint16_t ) ulTemp;
	}
	else
	{
		/* Make room for a 20 bit value stored as a 32 bit value. */
		pusTopOfStack = ( uint16_t * ) pxTopOfStack;		
		pusTopOfStack--;
		pulTopOfStack = ( uint32_t * ) pusTopOfStack;
		*pulTopOfStack = ( uint32_t ) pxCode;
	}

	pusTopOfStack--;
	*pusTopOfStack = portFLAGS_INT_ENABLED;
	pusTopOfStack -= ( sizeof( StackType_t ) / 2 );
	
	/* From here on the size of stacked items depends on the memory model. */
	pxTopOfStack = ( StackType_t * ) pusTopOfStack;

	/* Next the general purpose registers. */
	#ifdef PRELOAD_REGISTER_VALUES
		*pxTopOfStack = ( StackType_t ) 0xffff;
		pxTopOfStack--;
		*pxTopOfStack = ( StackType_t ) 0xeeee;
		pxTopOfStack--;
		*pxTopOfStack = ( StackType_t ) 0xdddd;
		pxTopOfStack--;
		*pxTopOfStack = ( StackType_t ) pvParameters;
		pxTopOfStack--;
		*pxTopOfStack = ( StackType_t ) 0xbbbb;
		pxTopOfStack--;
		*pxTopOfStack = ( StackType_t ) 0xaaaa;
		pxTopOfStack--;
		*pxTopOfStack = ( StackType_t ) 0x9999;
		pxTopOfStack--;
		*pxTopOfStack = ( StackType_t ) 0x8888;
		pxTopOfStack--;	
		*pxTopOfStack = ( StackType_t ) 0x5555;
		pxTopOfStack--;
		*pxTopOfStack = ( StackType_t ) 0x6666;
		pxTopOfStack--;
		*pxTopOfStack = ( StackType_t ) 0x5555;
		pxTopOfStack--;
		*pxTopOfStack = ( StackType_t ) 0x4444;
		pxTopOfStack--;
	#else
		pxTopOfStack -= 3;
		*pxTopOfStack = ( StackType_t ) pvParameters;
		pxTopOfStack -= 9;
	#endif

	/* A variable is used to keep track of the critical section nesting.
	This variable has to be stored as part of the task context and is
	initially set to zero. */
	*pxTopOfStack = ( StackType_t ) portNO_CRITICAL_SECTION_NESTING;	

	/* Return a pointer to the top of the stack we have generated so this can
	be stored in the task control block for the task. */
	return pxTopOfStack;
}
/*-----------------------------------------------------------*/

void vPortEndScheduler( void )
{
	/* It is unlikely that the MSP430 port will get stopped.  If required simply
	disable the tick interrupt here. */
}
/*-----------------------------------------------------------*/

/*
 * Hardware initialisation to generate the RTOS tick.
 */
void vPortSetupTimerInterrupt( void )
{
	vApplicationSetupTimerInterrupt();
}
/*-----------------------------------------------------------*/

long Tickcount = 0;
#pragma vector=configTICK_VECTOR
interrupt void vTickISREntry( void )
{
extern void vPortTickISR( void );

	__bic_SR_register_on_exit( SCG1 + SCG0 + OSCOFF + CPUOFF );
//    Tickcount++;
#ifdef CHECKPOINT
    exectued++;
    if(exectued >= PER)
    {
        exectued = 0;
        taskRecency[0] = timeCounter;
        memcpy(backup_sram, ((void*) 0x2C00),   0x1000);
    }
#endif
    timeCounter++;

#ifdef Meenchen
	/* Initialize variables */
	if(pxCurrentTCB->initial != 1){
	    pxCurrentTCB->StackInNVM = 1;
	    pxCurrentTCB->CodeInNVM = 1;
	    pxCurrentTCB->initial = 1;
	    pxCurrentTCB->AddressOfVMFunction = NULL;
	    pxCurrentTCB->CodeOffset = NULL;
	}

/*------------------------------  Extend to support dynamic caching: Start ------------------------------*/
	/*Write back current stack to FRAM: if we are using the stack in NVM, we should not(and not need to) mess up the "CURRENT" stack*/
	if(pxCurrentTCB->StackInNVM == 0){
//	    memcpy((void*) pxCurrentTCB->pxStack, (void*) pxCurrentTCB->AddressOfVMStack, configMINIMAL_STACK_SIZE * sizeof(StackType_t));
//	    StackToNVM = 1;
	}
/*------------------------------  Extend to support dynamic caching: End ------------------------------*/
#endif
	#if configUSE_PREEMPTION == 1
		extern void vPortPreemptiveTickISR( void );
		vPortPreemptiveTickISR();
	#else
		extern void vPortCooperativeTickISR( void );
		vPortCooperativeTickISR();
	#endif
#ifdef Meenchen
/*------------------------------  Extend to support dynamic caching: Start ------------------------------*/
	pxAddressOfNVMStack = pxCurrentTCB->pxStack;
	/* Restore to FRAM */
	if(pxCurrentTCB->StackInNVM == 0 && StackToNVM == 1){
	    /* Load the address and offset to registers */
	    asm(" MOV &pxAddressOfNVMStack, r12");
	    pxAddressOffset = pxCurrentTCB->AddressOffset;
	    asm(" MOV &pxAddressOffset, r13");
	    /* Change SP to the corresponding offset */
	    asm(" ADD r12, r13");
	    asm(" MOV r13, sp");
	    pxCurrentTCB->StackInNVM  = 1;
	}

	/* Caching stack to VM */
    if(pxCurrentTCB->StackInNVM == 1 && StackToNVM == 0){
        memcpy((void*) 0x1C00, (void*) pxCurrentTCB->pxStack, configMINIMAL_STACK_SIZE * sizeof(StackType_t) ); //0x1C00 ~ 0x1D54
        pxAddressOfVMStack = (void*)0x1C00; // should be the location allocated by our policy
        pxCurrentTCB->AddressOfVMStack = pxAddressOfVMStack;
        pxAddressOffset = (void*)0;
        /* Load the current sp*/
        asm(" MOV sp, r12");
        /* Load the stack address of VM*/
        asm(" MOV &pxAddressOfVMStack, r13");
        /* Load the stack address of NVM */
        asm(" MOV &pxAddressOfNVMStack, r14");
        /* Calculate the offset */
        asm(" SUB r14, r12");
        asm(" MOV r12, &pxAddressOffset");
        pxCurrentTCB->AddressOffset = pxAddressOffset;
        /* Change SP to the corresponding address of VM by the offset */
        asm(" ADD r12, r13");
        asm(" MOV r13, sp");
        pxCurrentTCB->StackInNVM = 0;
    }

#ifdef CacheCode
    /* Caching code to VM or Restore to NVM*/
    if(pxCurrentTCB->CodeInNVM != CodeToNVM){
        asm(" MOV 0x0014(sp), &temp");//At end of this function 1. POPM.A  #5,R15   2. RETI
        //get SR bits
        pxSR = (long)temp << 8;
        pxSR = (long)pxSR >> 8;
        //set prefix of PC
        temp = (long)temp >> 12;
        pxPC = (long)temp << 16;
        //add postfix of PC
        asm(" MOV 0x0016(sp), &temp");
        temp = (long)temp << 4;
        temp = (long)temp >> 4 ;
        pxPC = (long)pxPC + (long)temp;
        /* Caching to VM */
        if(pxCurrentTCB->CodeInNVM == 1 && CodeToNVM == 0){
            //offset of PC
            pxCurrentTCB->CodeOffset = pxPC - pxCurrentTCB->AddressOfNVMFunction;
            //copy code
            pxCurrentTCB->SizeOfFunction = 200;
            memcpy((void*) 0x1E00, pxCurrentTCB->AddressOfNVMFunction, pxCurrentTCB->SizeOfFunction);//0x1E00~0x1E50
            pxCurrentTCB->AddressOfVMFunction = (void*)0x1E00;
            pxPC = (long)pxCurrentTCB->AddressOfVMFunction + (long)pxCurrentTCB->CodeOffset;
            asm(" MOV &pxSR, 0x0014(sp)");
            asm(" MOV &pxPC, 0x0016(sp)");
            pxCurrentTCB->CodeInNVM = 0;
        }
        /* Restore to NVM */
        else if(pxCurrentTCB->CodeInNVM == 0 && CodeToNVM == 1){
            //offset of PC
            pxCurrentTCB->CodeOffset = pxPC - pxCurrentTCB->AddressOfVMFunction;
            pxPC = (long)pxCurrentTCB->AddressOfNVMFunction + (long)pxCurrentTCB->CodeOffset;
            //Set 20 bit of these two addresses
            temp = (long)pxPC >> 16;
            temp = (long)temp << 12;
            pxSR = (long)temp + (long)pxSR;
            temp = (long)pxPC << 4;
            pxPC = (long)temp >> 4;
            asm(" MOV &pxSR, 0x0014(sp)");
            asm(" MOV &pxPC, 0x0016(sp)");
            pxCurrentTCB->CodeInNVM = 1;
        }
    }
#endif
    /*------------------------------  Extend to support dynamic caching: End ------------------------------*/
#endif
}

/*-----------------------------------------------------------*/
	
