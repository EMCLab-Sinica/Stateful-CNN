/*
 * Overhead.c
 *
 *  Created on: 2018¦~5¤ë28¤é
 *      Author: Meenchen
 */
#include <stdio.h>
#include <DataManager/SimpDB.h>

void overhead()
{
    //create
    int ID = -1;
    struct working D;
    DBworking(&D, 2, ID);
    int* ptr = D.address;
    *ptr = 123;
    ID = DBcommit(&D,NULL,NULL,2,1);

    long count;
    int t = 1;

    //read
    for(count = 0; count < 1000000; count++){
        DBread(ID);
    }
    //copy on write
    for(count = 0; count < 1000000; count++){
        DBworking(&D, 2, ID);
        DBreadIn(&t, ID);
    }
    //commit
    for(count = 0; count < 1000000; count++){
        ID = DBcommit(&D,NULL,NULL,2,1);
    }
}
