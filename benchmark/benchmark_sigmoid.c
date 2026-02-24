#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

// Minimal setup for sigmoid params as in sigmoid.c
typedef struct {
    float white_target;
    float black_target;
    float paper_exp;
    float film_fog;
    float contrast_power;
    float skew_power;
} sigmoid_params;

void calculate_params(sigmoid_params* out) {
    // These are some realistic values derived from default settings
    out->white_target = 1.0f;
    out->black_target = 0.000152f;
    out->paper_exp = 0.5f;
    out->film_fog = 0.0f;
    out->contrast_power = 2.5f;
    out->skew_power = 1.0f;
}

void save_image(const char* filename, int width, int height, float* data) {
    unsigned char* pixels = (unsigned char*)malloc(width * height * 4);
    for (int i = 0; i < width * height * 4; i++) {
        float val = data[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        pixels[i] = (unsigned char)(val * 255.0f);
    }
    if (stbi_write_jpg(filename, width, height, 4, pixels, 90)) {
        printf("  Saved result to %s\n", filename);
    } else {
        printf("  Failed to save result to %s\n", filename);
    }
    free(pixels);
}

int main() {
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    int width = 6016;
    int height = 4016;
    size_t img_size = (size_t)width * height * 4 * sizeof(float);
    printf("Allocating %zu bytes for host data (%dx%d)...\n", img_size, width, height);
    float* h_data = (float*)malloc(img_size);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate memory for h_data\n");
        return 1;
    }
    printf("Initializing host data...\n");
    for (int y = 0; y < height; y++) {
        float h = (float)y / (height - 1) * 6.0f;
        int segment = (int)h;
        float f = h - segment;
        float pr, pg, pb;
        if (segment == 0) { pr = 1.0f; pg = f; pb = 0.0f; }
        else if (segment == 1) { pr = 1.0f - f; pg = 1.0f; pb = 0.0f; }
        else if (segment == 2) { pr = 0.0f; pg = 1.0f; pb = f; }
        else if (segment == 3) { pr = 0.0f; pg = 1.0f - f; pb = 1.0f; }
        else if (segment == 4) { pr = f; pg = 0.0f; pb = 1.0f; }
        else if (segment == 5) { pr = 1.0f; pg = 0.0f; pb = 1.0f - f; }
        else { pr = 1.0f; pg = 0.0f; pb = 0.0f; }

        for (int x = 0; x < width; x++) {
            float r, g, b;
            float mid_x = width / 2.0f;
            if (x < mid_x) {
                float t = x / mid_x;
                r = pr * t; g = pg * t; b = pb * t;
            } else {
                float t = (x - mid_x) / (width - 1.0f - mid_x);
                r = pr * (1.0f - t) + t;
                g = pg * (1.0f - t) + t;
                b = pb * (1.0f - t) + t;
            }
            int idx = (y * width + x) * 4;
            h_data[idx] = r;
            h_data[idx+1] = g;
            h_data[idx+2] = b;
            h_data[idx+3] = 1.0f;
        }
    }
    printf("Host data initialized.\n");

    printf("Initializing OpenCL...\n");
    // OpenCL Initialization
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found (err=%d)\n", err);
        return 1;
    }
    printf("Found %u OpenCL platform(s)\n", num_platforms);

    CHECK_CL(clGetPlatformIDs(1, &platform, NULL));
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
    
    char device_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Using device: %s\n", device_name);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err);
    
    // Create command queue with profiling enabled
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    if (err != CL_SUCCESS) {
        // Fallback for older OpenCL versions if needed, but we targeting 2.2 above
        queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    }
    CHECK_CL(err);

    // Load kernel source
    const char* kernel_path = "../data/kernels/sigmoid.cl";
    printf("Reading kernel source from %s...\n", kernel_path);
    char* source = read_file(kernel_path);
    if (!source) {
        fprintf(stderr, "Failed to load kernel source from %s\n", kernel_path);
        return 1;
    }

    // Build program
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
    CHECK_CL(err);
    
    // Need to point to the directory containing common.h and colorspace.h
    const char* options = "-I ../data/kernels/";
    err = clBuildProgram(program, 1, &device, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        char build_log[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
        fprintf(stderr, "Build error:\n%s\n", build_log);
        return 1;
    }

    printf("Creating kernels...\n");
    cl_kernel kernel_per_channel = clCreateKernel(program, "sigmoid_loglogistic_per_channel", &err);
    CHECK_CL(err);
    cl_kernel kernel_rgb_ratio = clCreateKernel(program, "sigmoid_loglogistic_rgb_ratio", &err);
    CHECK_CL(err);

    CHECK_CL(clGetPlatformIDs(1, &platform, &num_platforms));
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));

    cl_image_format format = { CL_RGBA, CL_FLOAT };
    cl_image_desc desc = { CL_MEM_OBJECT_IMAGE2D, width, height, 0, 0, 0, 0, 0, 0, {0} };
    
    printf("Creating OpenCL images...\n");
    cl_mem d_in = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, &desc, h_data, &err);
    CHECK_CL(err);
    cl_mem d_out = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, NULL, &err);
    CHECK_CL(err);

    printf("Creating OpenCL buffers for matrices...\n");
    float identity[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    cl_mem d_mat1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), identity, &err);
    CHECK_CL(err);
    cl_mem d_mat2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), identity, &err);
    CHECK_CL(err);
    cl_mem d_mat3 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), identity, &err);
    CHECK_CL(err);

    sigmoid_params p;
    calculate_params(&p);
    float hue_pres = 1.0f;

    printf("Saving input image to opencl_input.jpg...\n");
    save_image("opencl_input.jpg", width, height, h_data);

    // Benchmark Per-Channel Kernel
    printf("Benchmarking sigmoid_loglogistic_per_channel (Resolution: %dx%d)...\n", width, height);
    
    CHECK_CL(clSetKernelArg(kernel_per_channel, 0, sizeof(cl_mem), &d_in));
    CHECK_CL(clSetKernelArg(kernel_per_channel, 1, sizeof(cl_mem), &d_out));
    CHECK_CL(clSetKernelArg(kernel_per_channel, 2, sizeof(int), &width));
    CHECK_CL(clSetKernelArg(kernel_per_channel, 3, sizeof(int), &height));
    CHECK_CL(clSetKernelArg(kernel_per_channel, 4, sizeof(float), &p.white_target));
    CHECK_CL(clSetKernelArg(kernel_per_channel, 5, sizeof(float), &p.paper_exp));
    CHECK_CL(clSetKernelArg(kernel_per_channel, 6, sizeof(float), &p.film_fog));
    CHECK_CL(clSetKernelArg(kernel_per_channel, 7, sizeof(float), &p.contrast_power));
    CHECK_CL(clSetKernelArg(kernel_per_channel, 8, sizeof(float), &p.skew_power));
    CHECK_CL(clSetKernelArg(kernel_per_channel, 9, sizeof(float), &hue_pres));
    CHECK_CL(clSetKernelArg(kernel_per_channel, 10, sizeof(cl_mem), &d_mat1));
    CHECK_CL(clSetKernelArg(kernel_per_channel, 11, sizeof(cl_mem), &d_mat2));
    CHECK_CL(clSetKernelArg(kernel_per_channel, 12, sizeof(cl_mem), &d_mat3));

    int warmup_iters = 100;
    int iterations = 1000;
    cl_event event;
    double total_time = 0;

    printf("  Warmup (%d iterations)...\n", warmup_iters);
    for (int i = 0; i < warmup_iters; i++) {
        size_t global_work_size[2] = { width, height };
        CHECK_CL(clEnqueueNDRangeKernel(queue, kernel_per_channel, 2, NULL, global_work_size, NULL, 0, NULL, NULL));
    }
    clFinish(queue);

    printf("  Benchmarking (%d iterations)...\n", iterations);
    for (int i = 0; i < iterations; i++) {
        size_t global_work_size[2] = { width, height };
        CHECK_CL(clEnqueueNDRangeKernel(queue, kernel_per_channel, 2, NULL, global_work_size, NULL, 0, NULL, &event));
        clWaitForEvents(1, &event);
        
        cl_ulong start, end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
        total_time += (double)(end - start) / 1000000.0; // ns to ms
        clReleaseEvent(event);
    }
    printf("  Average Time: %.4f ms\n", total_time / iterations);

    // Save result of the last iteration to a file
    float* h_out = (float*)malloc(img_size);
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { (size_t)width, (size_t)height, 1 };
    CHECK_CL(clEnqueueReadImage(queue, d_out, CL_TRUE, origin, region, 0, 0, h_out, 0, NULL, NULL));

    save_image("opencl_output_per_channel.jpg", width, height, h_out);
    free(h_out);

    // Benchmark RGB Ratio Kernel
    printf("Benchmarking sigmoid_loglogistic_rgb_ratio (Resolution: %dx%d)...\n", width, height);

    CHECK_CL(clSetKernelArg(kernel_rgb_ratio, 0, sizeof(cl_mem), &d_in));
    CHECK_CL(clSetKernelArg(kernel_rgb_ratio, 1, sizeof(cl_mem), &d_out));
    CHECK_CL(clSetKernelArg(kernel_rgb_ratio, 2, sizeof(int), &width));
    CHECK_CL(clSetKernelArg(kernel_rgb_ratio, 3, sizeof(int), &height));
    CHECK_CL(clSetKernelArg(kernel_rgb_ratio, 4, sizeof(float), &p.white_target));
    CHECK_CL(clSetKernelArg(kernel_rgb_ratio, 5, sizeof(float), &p.black_target));
    CHECK_CL(clSetKernelArg(kernel_rgb_ratio, 6, sizeof(float), &p.paper_exp));
    CHECK_CL(clSetKernelArg(kernel_rgb_ratio, 7, sizeof(float), &p.film_fog));
    CHECK_CL(clSetKernelArg(kernel_rgb_ratio, 8, sizeof(float), &p.contrast_power));
    CHECK_CL(clSetKernelArg(kernel_rgb_ratio, 9, sizeof(float), &p.skew_power));

    printf("  Warmup (%d iterations)...\n", warmup_iters);
    for (int i = 0; i < warmup_iters; i++) {
        size_t global_work_size[2] = { width, height };
        CHECK_CL(clEnqueueNDRangeKernel(queue, kernel_rgb_ratio, 2, NULL, global_work_size, NULL, 0, NULL, NULL));
    }
    clFinish(queue);

    total_time = 0;
    printf("  Benchmarking (%d iterations)...\n", iterations);
    for (int i = 0; i < iterations; i++) {
        size_t global_work_size[2] = { width, height };
        CHECK_CL(clEnqueueNDRangeKernel(queue, kernel_rgb_ratio, 2, NULL, global_work_size, NULL, 0, NULL, &event));
        clWaitForEvents(1, &event);
        
        cl_ulong start, end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
        total_time += (double)(end - start) / 1000000.0; // ns to ms
        clReleaseEvent(event);
    }
    printf("  Average Time: %.4f ms\n", total_time / iterations);

    // Save result of the last iteration to a file
    h_out = (float*)malloc(img_size);
    CHECK_CL(clEnqueueReadImage(queue, d_out, CL_TRUE, origin, region, 0, 0, h_out, 0, NULL, NULL));

    save_image("opencl_output_rgb_ratio.jpg", width, height, h_out);
    free(h_out);

    // Cleanup
    free(source);
    free(h_data);
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseMemObject(d_mat1);
    clReleaseMemObject(d_mat2);
    clReleaseMemObject(d_mat3);
    clReleaseKernel(kernel_per_channel);
    clReleaseKernel(kernel_rgb_ratio);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
