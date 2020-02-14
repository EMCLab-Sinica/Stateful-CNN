#pragma once

#include <msp430.h> /* __no_operation() */

#define ERROR_OCCURRED() for (;;) { __no_operation(); }
