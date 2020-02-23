#include <FreeRTOS.h>
#include <list.h>
#include <stdio.h>

static const size_t LIST_SIZE = 5;

int main(void) {
    List_t list;
    ListItem_t items[LIST_SIZE];
    int values[LIST_SIZE];

    vListInitialise(&list);
    for (size_t i = 0; i < LIST_SIZE; i++) {
        vListInitialiseItem(items + i);
        values[i] = i;
        listSET_LIST_ITEM_OWNER(items + i, values + i);
        vListInsertEnd(&list, items + i);
    }

    int *valptr = NULL;
    for (size_t i = 0; i < LIST_SIZE * 2; i++) {
        listGET_OWNER_OF_NEXT_ENTRY(valptr, &list);
        printf("%d ", *valptr);
    }

    printf("\n");

    return 0;
}
