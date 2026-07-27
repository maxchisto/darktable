#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define CHECK_CL(cmd) \
    { \
        cl_int _cl_err = cmd; \
        if (_cl_err != CL_SUCCESS) { \
            fprintf(stderr, "OpenCL error %d at %s:%d\n", _cl_err, __FILE__, __LINE__); \
            exit(1); \
        } \
    }

char* read_file(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(size + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, size, f) != (size_t)size) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[size] = '\0';
    fclose(f);
    return buf;
}

static double run_benchmark(cl_context context, cl_command_queue queue, cl_program program,
                            cl_mem d_in, int width, int height, int radius) {
    int k_width = 2 * radius + 1;
    int iterations = 100;
    if (radius >= 12) iterations = 30;
    else if (radius >= 5) iterations = 80;

    // Create kernel + output image (reusable)
    cl_image_format format = { CL_RGBA, CL_FLOAT };
    cl_image_desc desc = { CL_MEM_OBJECT_IMAGE2D, width, height, 0, 0, 0, 0, 0, 0, {NULL} };
    cl_int err;
    cl_mem d_out = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, NULL, &err);
    CHECK_CL(err);

    // Create kernel image for this radius
    size_t k_data_size = (size_t)k_width * k_width * sizeof(float);
    float* h_kern = (float*)malloc(k_data_size);
    for (int i = 0; i < k_width * k_width; i++)
        h_kern[i] = 1.0f / (k_width * k_width);

    cl_image_format kern_format = { CL_R, CL_FLOAT };
    cl_image_desc kern_desc = { CL_MEM_OBJECT_IMAGE2D, k_width, k_width, 0, 0, 0, 0, 0, 0, {NULL} };
    cl_mem d_kern = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  &kern_format, &kern_desc, h_kern, &err);
    CHECK_CL(err);

    cl_kernel kernel = clCreateKernel(program, "convolve", &err);
    CHECK_CL(err);

    CHECK_CL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
    CHECK_CL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_kern));
    CHECK_CL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_out));
    CHECK_CL(clSetKernelArg(kernel, 3, sizeof(int), &width));
    CHECK_CL(clSetKernelArg(kernel, 4, sizeof(int), &height));
    CHECK_CL(clSetKernelArg(kernel, 5, sizeof(int), &radius));

    printf("Benchmarking OpenCL convolve (Radius: %d, %dx%d, %d iters)...\n", radius, width, height, iterations);
    fflush(stdout);

    cl_event event;
    double total_time = 0;
    for (int i = 0; i < iterations; i++) {
        size_t global_work_size[2] = { (size_t)width, (size_t)height };
        CHECK_CL(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, &event));
        clWaitForEvents(1, &event);
        cl_ulong start, end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
        total_time += (double)(end - start) / 1000000.0;
        clReleaseEvent(event);
    }
    double avg = total_time / iterations;
    printf("  Average Time: %.4f ms\n", avg);

    clReleaseMemObject(d_kern);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    free(h_kern);
    return avg;
}

int main() {
    printf("Starting C benchmark...\n");
    fflush(stdout);

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    int width = 6016;
    int height = 4016;
    size_t img_size = (size_t)width * height * 4 * sizeof(float);

    printf("Allocating memory for %dx%d image...\n", width, height);
    fflush(stdout);
    float* h_data = (float*)malloc(img_size);
    if (!h_data) { fprintf(stderr, "Failed to allocate h_data\n"); return 1; }

    printf("Initializing host data (%zu bytes)...\n", img_size);
    fflush(stdout);
    for (size_t i = 0; i < (size_t)width * height * 4; i++) h_data[i] = 0.5f;
    printf("Host data initialized.\n");
    fflush(stdout);

    // OpenCL Initialization
    printf("Initializing OpenCL...\n");
    fflush(stdout);
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found\n");
        return 1;
    }
    CHECK_CL(clGetPlatformIDs(1, &platform, NULL));
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));

    char device_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Using device: %s\n", device_name);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err);
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err);

    // Load kernel source
    const char* kernel_path = "../../data/kernels/blurs.cl";
    char* source = read_file(kernel_path);
    if (!source) { fprintf(stderr, "Failed to load kernel\n"); return 1; }

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
    CHECK_CL(err);
    const char* options = "-I ../../data/kernels/";
    err = clBuildProgram(program, 1, &device, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        return 1;
    }

    // Create input image (reused across radii)
    cl_image_format format = { CL_RGBA, CL_FLOAT };
    cl_image_desc desc = { CL_MEM_OBJECT_IMAGE2D, width, height, 0, 0, 0, 0, 0, 0, {NULL} };
    cl_mem d_in = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                &format, &desc, h_data, &err);
    CHECK_CL(err);

    int radii[] = {3, 8, 15};
    int num_radii = sizeof(radii) / sizeof(radii[0]);
    for (int r = 0; r < num_radii; r++) {
        run_benchmark(context, queue, program, d_in, width, height, radii[r]);
    }

    free(source); free(h_data);
    clReleaseMemObject(d_in);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
