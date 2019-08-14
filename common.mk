CPPFLAGS = -I .
DEBUG = 0
CFLAGS = -std=c99 -Wall -Wextra -Wstrict-prototypes -Wconversion -Wshadow
ifeq ($(DEBUG),1)
    CFLAGS += -g -O0
else
    CFLAGS += -O3
    CPPFLAGS += -DNDEBUG
endif

clean-common:
	rm -rf *.o *.dSYM __pycache__
