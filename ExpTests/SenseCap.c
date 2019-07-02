/*
 * SenseCap.c
 *
 *  Created on: 2018/3/13
 *      Author: Meenchen
 */
#include <FreeRTOS.h>
#include <task.h>
#include <driverlib.h>
#include <DataManager/SimpDB.h>
#include "SenseCap.h"
#include "Semph.h"
#include <config.h>

int Cbuffersize = 5;
int Cptr;
int Ccircled;
extern int waitCap;
extern int ADCSemph;
extern int capID;
extern unsigned long information[10];
extern int readCap;

void CapLog(){
    while(1)
    {
        acquireSemph();
        //Initialize for validity interval
        registerTCB(IDCAP);

        //Initialize the ADC12B Module
        ADC12_B_initParam initParam = {0};
        initParam.sampleHoldSignalSourceSelect = ADC12_B_SAMPLEHOLDSOURCE_SC;
        initParam.clockSourceSelect = ADC12_B_CLOCKSOURCE_ACLK;
        initParam.clockSourceDivider = ADC12_B_CLOCKDIVIDER_1;
        initParam.clockSourcePredivider = ADC12_B_CLOCKPREDIVIDER__1;
        initParam.internalChannelMap = ADC12_B_BATTMAP;
        ADC12_B_init(ADC12_B_BASE, &initParam);

        // Enable the ADC12B module
        ADC12_B_enable(ADC12_B_BASE);

        // Sets up the sampling timer pulse mode
        ADC12_B_setupSamplingTimer(ADC12_B_BASE,
                                   ADC12_B_CYCLEHOLD_128_CYCLES,
                                   ADC12_B_CYCLEHOLD_128_CYCLES,
                                   ADC12_B_MULTIPLESAMPLESDISABLE);

        // Maps Temperature Sensor input channel to Memory 0 and select voltage references
        ADC12_B_configureMemoryParam configureMemoryParam = {0};
        configureMemoryParam.memoryBufferControlIndex = ADC12_B_MEMORY_0;
        configureMemoryParam.inputSourceSelect = ADC12_B_INPUT_BATMAP;//ADC12_B_INPUT_BATMAP
        configureMemoryParam.refVoltageSourceSelect =
            ADC12_B_VREFPOS_INTBUF_VREFNEG_VSS;
        configureMemoryParam.endOfSequence = ADC12_B_NOTENDOFSEQUENCE;
        configureMemoryParam.windowComparatorSelect =
            ADC12_B_WINDOW_COMPARATOR_DISABLE;
        configureMemoryParam.differentialModeSelect =
            ADC12_B_DIFFERENTIAL_MODE_DISABLE;
        ADC12_B_configureMemory(ADC12_B_BASE, &configureMemoryParam);

        // Clear memory buffer 0 interrupt
        ADC12_B_clearInterrupt(ADC12_B_BASE,
                               0,
                               ADC12_B_IFG0
                               );

        // Enable memory buffer 0 interrupt
        ADC12_B_enableInterrupt(ADC12_B_BASE,
                                ADC12_B_IE0,
                                0,
                                0);

        // Configure internal reference
        while(Ref_A_isRefGenBusy(REF_A_BASE));              // If ref generator busy, WAIT
        Ref_A_enableTempSensor(REF_A_BASE);
        Ref_A_setReferenceVoltage(REF_A_BASE, REF_A_VREF2_0V);
        Ref_A_enableReferenceVoltage(REF_A_BASE);

        /*
         * Base address of ADC12B Module
         * Start the conversion into memory buffer 0
         * Use the single-channel, single-conversion mode
         */
        waitCap = 1;
        ADC12_B_startConversion(ADC12_B_BASE,
                                ADC12_B_MEMORY_0,
                                ADC12_B_SINGLECHANNEL);
        __bis_SR_register(GIE);   // Wait for conversion to complete
        while(waitCap){

        }
        __bic_SR_register(GIE);

        readCap++;
        // Disable ADC12
//        ADC12_B_disable(ADC12_B_BASE);
        ADCSemph = 0;
        //write the result
        struct working data;
        DBworking(&data, 2, capID);//2 bytes for int and for create(-1)
        int* capadd = data.address;
        *capadd = ADC12MEM0;
        capID = DBcommit(&data,NULL,NULL,2,1);
        //validity done
        information[IDCAP]++;
        unresgisterTCB(IDCAP);
        portYIELD();
    }
}

