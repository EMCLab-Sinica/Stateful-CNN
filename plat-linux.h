#pragma once

#include <stdio.h>
#include <inttypes.h>
#include <signal.h>

#define ERROR_OCCURRED() do { raise(SIGINT); } while (0);
