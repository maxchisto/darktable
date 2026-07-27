#include <CL/cl.h>
#include <stdio.h>

int main() {
    printf("Checking OpenCL platforms...\n");
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS) {
        printf("clGetPlatformIDs failed with %d\n", err);
        return 1;
    }
    printf("Found %u platforms.\n", num_platforms);
    return 0;
}
