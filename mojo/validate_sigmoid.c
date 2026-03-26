#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <stdint.h>

// Mock the darktable module structure minimally
typedef struct dt_iop_module_so_t {
  void *data;
} dt_iop_module_so_t;

typedef void (*init_global_fn)(dt_iop_module_so_t *);
typedef void (*cleanup_global_fn)(dt_iop_module_so_t *);

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_to_libsigmoid.so>\n", argv[0]);
        return 1;
    }

    const char *plugin_path = argv[1];
    printf("--- Validating Plugin: %s ---\n", plugin_path);

    void *handle = dlopen(plugin_path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        fprintf(stderr, "FAILED to dlopen %s: %s\n", plugin_path, dlerror());
        return 1;
    }
    printf("SUCCESS: Plugin loaded.\n");

    init_global_fn init_global = (init_global_fn)dlsym(handle, "init_global");
    cleanup_global_fn cleanup_global = (cleanup_global_fn)dlsym(handle, "cleanup_global");

    if (!init_global) {
        fprintf(stderr, "FAILED: Could not find symbol 'init_global'\n");
        return 1;
    }

    printf("SUCCESS: Found 'init_global'. Calling it now...\n");

    dt_iop_module_so_t so = { .data = NULL };
    
    // This will trigger the dlopen for libsigmoid_mojo.so
    init_global(&so);

    if (so.data == NULL) {
        printf("FAILED: init_global did not set so.data. Mojo library likely failed to load or dlsym failed.\n");
    } else {
        printf("SUCCESS: init_global executed and so.data = %p\n", so.data);
        
        // We can't easily peek into the opaque struct without the header,
        // but we can check if cleanup works.
        if (cleanup_global) {
            printf("Calling cleanup_global...\n");
            cleanup_global(&so);
            printf("SUCCESS: cleanup_global executed.\n");
        }
    }

    dlclose(handle);
    printf("--- Validation Complete ---\n");
    return 0;
}
