/*
 * dvfs.h
 *
 *  Created on: 2016/1/26
 */

#ifndef DVFS_H_
#define DVFS_H_

#ifdef __cplusplus
extern "C" {
#endif

extern unsigned int FreqLevel;

void setFrequency(int level);
unsigned long getFrequency(int level);

#ifdef __cplusplus
}
#endif

#endif /* DVFS_H_ */
