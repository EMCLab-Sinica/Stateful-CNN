#include <driverlib.h> // for timer interrupt names
#include "my_timer.h"
#include "FreeRTOSConfig.h"

uint32_t tickCounter = 0;

#pragma vector=configTICK_VECTOR
__interrupt void vTimerHandler( void )
{
    tickCounter++;
}

uint32_t getTickCounter( void )
{
    return tickCounter;
}
