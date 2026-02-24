#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h> // The main OpenCL header containing all types and functions
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* 
 * OpenCL is a "host-controlled" API. This means the CPU (Host) explicitly 
 * manages the GPU (Device). Almost every OpenCL function returns an error code. 
 * This macro helps us ensure everything succeeded before moving to the next step.
 */
#define CHECK_CL(cmd) \
    { \
        cl_int err = cmd; \
        if (err != CL_SUCCESS) { \
            fprintf(stderr, "OpenCL error %d at %s:%d\n", err, __FILE__, __LINE__); \
            exit(1); \
        } \
    }

/* 
 * Helper function to read the OpenCL kernel source code from a file.
 * OpenCL kernels are usually compiled at runtime from source strings.
 */
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
    fread(buf, 1, size, f);
    buf[size] = '\0';
    fclose(f);
    return buf;
}

/* 
 * This C-struct MUST match the memory layout expected by the OpenCL kernel 
 * if we were passing the entire struct as an argument (though here we pass fields individually).
 */
typedef struct {
    float white_target;
    float black_target;
    float paper_exp;
    float film_fog;
    float contrast_power;
    float skew_power;
} sigmoid_params;

int main() {
    /* 
     * STEP 1: Discover and Initialize the OpenCL Environment
     * This is the "boilerplate" required to connect the CPU to the GPU.
     */
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int err;

    // 1a. Find an OpenCL platform (e.g., NVIDIA, AMD, Intel)
    CHECK_CL(clGetPlatformIDs(1, &platform, NULL));
    
    // 1b. Find a GPU device on that platform
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
    
    // 1c. Create a context: an environment where kernels and memory objects live
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err);
    
    // 1d. Create a command queue: the "conveyor belt" where we send tasks to the GPU
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, 0, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    CHECK_CL(err);

    /* 
     * STEP 2: Compile the GPU Kernel
     * Unlike standard C code, GPU code is compiled during program execution 
     * to ensure it's optimized for the specific GPU hardware present.
     */
    const char* kernel_path = "../data/kernels/sigmoid.cl";
    char* source = read_file(kernel_path);
    
    // 2a. Create the program object from the source string
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
    CHECK_CL(err);
    
    // 2b. Compile (Build) the program for our specific GPU device
    const char* options = "-I ../data/kernels/";
    err = clBuildProgram(program, 1, &device, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        // If compilation fails, we must check the Build Log for syntax errors
        char build_log[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
        fprintf(stderr, "Build error:\n%s\n", build_log);
        return 1;
    }

    // 2c. Extract specific functions (kernels) from the compiled program
    cl_kernel kernel_rgb_ratio = clCreateKernel(program, "sigmoid_loglogistic_rgb_ratio", &err);
    CHECK_CL(err);
    cl_kernel kernel_per_channel = clCreateKernel(program, "sigmoid_loglogistic_per_channel", &err);
    CHECK_CL(err);

    /* 
     * STEP 3: Setup Host Memory (CPU side)
     * We create a synthetic test image (a gradient) to verify the math parity.
     */
    int width = 6016;
    int height = 4016;
    size_t img_size = width * height * 4 * sizeof(float);
    float* h_in = (float*)malloc(img_size);
    float* h_out = (float*)malloc(img_size);
    
    // Generate a colorful gradient for testing
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
            h_in[idx] = r;
            h_in[idx+1] = g;
            h_in[idx+2] = b;
            h_in[idx+3] = 1.0f;
        }
    }

    /* 
     * STEP 4: Setup Device Memory (GPU side)
     * The GPU cannot directly read CPU RAM. We must allocate space on the GPU 
     * and copy the data over.
     */
    
    // Define the image format (RGBA floats) and descriptor (2D image dimensions)
    cl_image_format format = { CL_RGBA, CL_FLOAT };
    cl_image_desc desc = { CL_MEM_OBJECT_IMAGE2D, width, height, 0, 0, 0, 0, 0, 0, {0} };
    
    // 4a. Create GPU "Image" objects. Images are optimized for 2D spatial locality caches.
    // CL_MEM_COPY_HOST_PTR tells OpenCL to copy h_in to the GPU immediately.
    cl_mem d_in = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, &desc, h_in, &err);
    CHECK_CL(err);
    cl_mem d_out = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, NULL, &err);
    CHECK_CL(err);

    // 4b. Create standard "Buffer" objects for extra data (identity matrices)
    float identity[16] = {0};
    identity[0] = 1; identity[5] = 1; identity[10] = 1; identity[15] = 1;
    cl_mem d_mat1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), identity, &err);
    CHECK_CL(err);
    cl_mem d_mat2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), identity, &err);
    CHECK_CL(err);
    cl_mem d_mat3 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16 * sizeof(float), identity, &err);
    CHECK_CL(err);

    // Algorithm parameters
    sigmoid_params p = { 1.0f, 0.000152f, 0.5f, 0.0f, 2.5f, 1.0f };
    float hue_pres = 1.0f;

    /* 
     * STEP 5: Execute Kernels
     * We must "bind" the arguments to the kernel before telling the GPU to run it.
     */
    
    // 5a. Set arguments for the first kernel (RGB Ratio)
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

    // 5b. Define the grid size. We want 1 GPU thread per pixel.
    size_t global_work_size[2] = { width, height };
    
    // 5c. Launch the kernel! This is non-blocking; it puts the task into the queue.
    CHECK_CL(clEnqueueNDRangeKernel(queue, kernel_rgb_ratio, 2, NULL, global_work_size, NULL, 0, NULL, NULL));
    
    // 5d. Retrieve the results back from GPU to CPU memory.
    // CL_TRUE makes this a blocking call: we wait until the results are actually copied back.
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { width, height, 1 };
    CHECK_CL(clEnqueueReadImage(queue, d_out, CL_TRUE, origin, region, 0, 0, h_out, 0, NULL, NULL));

    // Print some pixels to check results
    printf("Parity Check (Input: Gradient)\n");
    int coords[3][2] = {
        {(int)(width * 0.2), (int)(height * 0.2)},
        {(int)(width * 0.4), (int)(height * 0.4)},
        {(int)(width * 0.6), (int)(height * 0.6)}
    };

    for (int i = 0; i < 3; i++) {
        int x = coords[i][0];
        int y = coords[i][1];
        int idx = (y * width + x) * 4;
        printf("RGB Ratio - %d%% pixel: [%.8f, %.8f, %.8f, %.8f]\n", (i+1)*20, h_out[idx], h_out[idx+1], h_out[idx+2], h_out[idx+3]);
    }

    // 5e. Repeat for the second kernel (Per Channel)
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

    CHECK_CL(clEnqueueNDRangeKernel(queue, kernel_per_channel, 2, NULL, global_work_size, NULL, 0, NULL, NULL));
    CHECK_CL(clEnqueueReadImage(queue, d_out, CL_TRUE, origin, region, 0, 0, h_out, 0, NULL, NULL));
    
    for (int i = 0; i < 3; i++) {
        int x = coords[i][0];
        int y = coords[i][1];
        int idx = (y * width + x) * 4;
        printf("Per Channel - %d%% pixel: [%.8f, %.8f, %.8f, %.8f]\n", (i+1)*20, h_out[idx], h_out[idx+1], h_out[idx+2], h_out[idx+3]);
    }

    /* 
     * STEP 6: Cleanup
     * OpenCL objects are reference-counted. We must release them 
     * to avoid memory leaks on both the CPU and the GPU.
     */
    free(source);
    free(h_in);
    free(h_out);
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseMemObject(d_mat1);
    clReleaseMemObject(d_mat2);
    clReleaseMemObject(d_mat3);
    clReleaseKernel(kernel_rgb_ratio);
    clReleaseKernel(kernel_per_channel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
