/*
 * config.h
 *
 *  Created on: 2018¦~2¤ë13¤é
 *      Author: Meenchen
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#define DATACONSISTENCY 1
//#define ONNVM //read/write on NVM
//#define ONVM // read/write on VM: need to copy all data once require to read after power resumes
#define OUR //read from NVM/VM, write to VM
//#define ONEVERSION //all read on NVM/commit to NVM

#define NUMTASK 12
#define NUMDATA 40
#define MAXREAD 5

//Used for TI applications
#define ITERFIR 1
#define ITERMATH16 500
#define ITERMATH32 300
#define ITER2DMATRIX 20
#define ITERMATRIXMUL 10

#define IDCAP 0
#define IDTEMP 1
#define IDCAPCALIBRATE 2
#define IDTEMPCALIBRATE 3
#define IDUART 4
#define IDFIR 5
#define IDMATH16 6
#define IDMATH32 7
#define ID2DMATRIX 8
#define IDMATRIXMUL 9

#endif /* CONFIG_H_ */
