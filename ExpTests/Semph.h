/*
 * Semph.h
 *
 *  Created on: 2018/3/18
 *      Author: Meenchen
 */

#ifndef EXPTESTS_SEMPH_H_
#define EXPTESTS_SEMPH_H_

#include "task.h"

extern int ADCSemph;

static void acquireSemph()
{
    taskENTER_CRITICAL();
    while(ADCSemph != 0)portYIELD();
    ADCSemph++;
    taskEXIT_CRITICAL();
}

static void releaseSemph()
{
    taskENTER_CRITICAL();
    ADCSemph++;
    taskEXIT_CRITICAL();
}

#endif /* EXPTESTS_SEMPH_H_ */
