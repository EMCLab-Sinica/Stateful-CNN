/*
 * dvfs.h
 *
 *  Created on: 2016/1/26
 *      Author: WeiMingChen
 */

#ifndef DVFS_H_
#define DVFS_H_

extern unsigned int FreqLevel;

void setFrequency(int level);
unsigned long getFrequency(int level);


#endif /* DVFS_H_ */
