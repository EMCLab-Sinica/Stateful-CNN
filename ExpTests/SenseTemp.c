/*
 * SenseTemp.c
 *
 *  Created on: 2018¦~3¤ë13¤é
 *      Author: Meenchen
 */
#define CALADC12_12V_30C  *((unsigned int *)0x1A1A)   // Temperature Sensor Calibration-30 C
#define CALADC12_12V_85C  *((unsigned int *)0x1A1C)   // Temperature Sensor Calibration-85 C

#include <FreeRTOS.h>
#include <task.h>
#include <driverlib.h>
#include <DataManager/SimpDB.h>
#include "SenseTemp.h"
#include "Semph.h"
#include <config.h>

int buffersize = 5;
int ptr;
int circled;
extern int waitTemp;
extern int ADCSemph;
extern int tempID;
extern unsigned long information[10];
extern int readTemp;

void SenseLog(){
    while(1)
    {
        acquireSemph();
        //Initialize for validity interval
        registerTCB(IDTEMP);

        //Initialize the ADC12B Module
        ADC12_B_initParam initParam = {0};
        initParam.sampleHoldSignalSourceSelect = ADC12_B_SAMPLEHOLDSOURCE_SC;
        initParam.clockSourceSelect = ADC12_B_CLOCKSOURCE_ACLK;
        initParam.clockSourceDivider = ADC12_B_CLOCKDIVIDER_1;
        initParam.clockSourcePredivider = ADC12_B_CLOCKPREDIVIDER__1;
        initParam.internalChannelMap = ADC12_B_TEMPSENSEMAP;
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
        configureMemoryParam.memoryBufferControlIndex = ADC12_B_MEMORY_1;
        configureMemoryParam.inputSourceSelect = ADC12_B_INPUT_TCMAP;
        configureMemoryParam.refVoltageSourceSelect =
            ADC12_B_VREFPOS_INTBUF_VREFNEG_VSS;
        configureMemoryParam.endOfSequence = ADC12_B_NOTENDOFSEQUENCE;
        configureMemoryParam.windowComparatorSelect =
            ADC12_B_WINDOW_COMPARATOR_DISABLE;
        configureMemoryParam.differentialModeSelect =
            ADC12_B_DIFFERENTIAL_MODE_DISABLE;
        ADC12_B_configureMemory(ADC12_B_BASE, &configureMemoryParam);

        // Clear memory buffer 1 interrupt
        ADC12_B_clearInterrupt(ADC12_B_BASE,
                               0,
                               ADC12_B_IFG2
                               );

        // Enable memory buffer 1 interrupt
        ADC12_B_enableInterrupt(ADC12_B_BASE,
                                ADC12_B_IE2,
                                0,
                                0);


        // Configure internal reference
        while(Ref_A_isRefGenBusy(REF_A_BASE));              // If ref generator busy, WAIT
        Ref_A_enableTempSensor(REF_A_BASE);
        Ref_A_setReferenceVoltage(REF_A_BASE, REF_A_VREF1_2V);
        Ref_A_enableReferenceVoltage(REF_A_BASE);

        /*
         * Base address of ADC12B Module
         * Start the conversion into memory buffer 0
         * Use the single-channel, single-conversion mode
         */
        waitTemp = 1;
        ADC12_B_startConversion(ADC12_B_BASE,
                                ADC12_B_MEMORY_1,
                                ADC12_B_SINGLECHANNEL);
        __bis_SR_register(GIE);   // Wait for conversion to complete
        while(waitTemp){

        }
        __bic_SR_register(GIE);

        //Write to buffer
        int temp = (int)ADC12MEM1;

        readTemp++;
        ADCSemph = 0;
        //write the result
        struct working data;
        DBworking(&data, 4, tempID);//2 bytes for int and for create(-1)
        float* tempadd = data.address;
        *tempadd = (float)(((long)temp - CALADC12_12V_30C) * (85 - 30)) / (CALADC12_12V_85C - CALADC12_12V_30C) + 30.0f;
        tempID = DBcommit(&data,NULL,NULL,4,1);
        information[IDTEMP]++;
        unresgisterTCB(IDTEMP);
        portYIELD();
    }
}


