#pragma once

#include "platform.h"

// In TI-DSPLib, __no_operation is called when msp_checkStatus failed, so also make it fail on PC
#define __no_operation ERROR_OCCURRED
