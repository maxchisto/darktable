#include <stdio.h>
#include <dlfcn.h>

typedef void (*launch_fn)();

int main() {
    void* handle = dlopen("./lib_fault.so", RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    launch_fn run_fail = (launch_fn)dlsym(handle, "run_failing_case");
    launch_fn run_work = (launch_fn)dlsym(handle, "run_working_case");

    printf("--- Running WORKING Case (Indirection) ---\n");
    run_work();
    printf("C: WORKING case worked.\n\n");

    printf("--- Running FAILING Case (Direct) ---\n");
    run_fail();
    printf("C: FAILING case worked (Unexpected).\n");

    dlclose(handle);
    return 0;
}
