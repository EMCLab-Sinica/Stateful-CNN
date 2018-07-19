/*
 * Recover.h
 *
 *  Created on: 2018¦~2¤ë12¤é
 *      Author: Meenchen
 */

#include <config.h>

#ifndef RECOVERYHANDLER_RECOVERY_H_
#define RECOVERYHANDLER_RECOVERY_H_

void taskRerun();

/* Used for rerunning unfinished tasks */
#pragma NOINIT(unfinished)
static unsigned short unfinished[NUMTASK];// 1: running, others for invalid
#pragma NOINIT(address)
static void* address[NUMTASK];// Function address of tasks
#pragma NOINIT(priority)
static unsigned short priority[NUMTASK];
#pragma NOINIT(TCBNum)
static unsigned short TCBNum[NUMTASK];
#pragma NOINIT(TCBAdd)
static void* TCBAdd[NUMTASK];// TCB address of tasks
#pragma NOINIT(schedulerTask)
static int schedulerTask[NUMTASK];// if it is schduler's task, we don't need to recreate it because the shceduler does


void taskRerun();
void regTaskStart(void* add, unsigned short pri, unsigned short TCB, void* TCBA, int stopTrack);
void regTaskEnd();
void failureRecovery();
void freePreviousTasks();

#endif /* RECOVERYHANDLER_RECOVERY_H_ */
