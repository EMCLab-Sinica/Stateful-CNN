/*
 * SimpDB.h
 *
 *  Created on: 2017¦~7¤ë12¤é
 *      Author: Meenchen
 *  Description: This simple DB is used to manage data and task snapshot(stacks)
 */
#include <../DataManager/maps.h>
#include <FreeRTOSConfig.h>
#include <stdint.h>
#include <config.h>

#define TOTAL_DATA_SIZE 16 * NUMOBJ
#define DWORKSIZE 3072
#define NVM 0
#define VM 1

#ifdef DATACONSISTENCY
// used for validation: Task t with Task's TCB = WSRTCB[i], SRBegin[NUMTASK] = min(writer's begin), WSRValid[NUMTASK] = 1
static unsigned long WSRBegin[NUMTASK]; //The "begin time of every commit operation" for an object "read by task i" is saved in WSRBegin[i]
static unsigned short WSRTCB[NUMTASK];
static unsigned char WSRValid[NUMTASK];
#endif

struct working{//working space of data for tasks
    void* address;
    int loc;//1 stands for SRAM, 0 stands for¡@NVM
    int id;//-1 for create
};

struct data{//two-version data structure
    void* cacheAdd;//Should point to VM or NVM(depends on mode)
    unsigned char cacheLoc;//indicate the location of the cached address
    unsigned int size;
    unsigned short readTCBNum[MAXREAD];//store 5 readers' TCB number
};

/* for validation */
extern unsigned long timeCounter;

static int DBmode = VM; //indicate where should the working copy and commit data be

#pragma NOINIT(DBSpace) //space for maintaining data structure of data
static uint8_t DBSpace[TOTAL_DATA_SIZE];

#pragma NOINIT(DB) //data structures for all data
static struct data* DB;

#pragma DATA_SECTION(dataId, ".map") //id for data labeling
static int dataId;

/* Half of RAM for caching (0x2C00~0x3800) */
#pragma location = 0x2C00 //Space for working at SRAM
static unsigned char SRAMData[DWORKSIZE];

#pragma DATA_SECTION(DWORKpoint, ".map") //RR pointer for working at SRAM
static unsigned int DWORKpoint;

/* DB functions */
void constructor();
void destructor();
int DBcommit(struct working *work, struct working *work2, struct working *work3, int size, int num);
void* DBread(int id);
void DBreadIn(void* to,int id);
void DBworking(struct working* wIn, int size, int id);
void DBmodeSelect(int select);

/* functions for validation*/
void registerTCB(int id);
void unresgisterTCB(int id);

/* internal functions */
static unsigned long min(unsigned long a, unsigned long b){
    if (a > b)
        return b;
    else
        return a;
}

