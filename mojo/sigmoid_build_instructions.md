# Darktable Sigmoid Mojo Build Instructions

This guide outlines how to compile the Mojo-based Sigmoid module and its C-bridge for integration with darktable.

## 1. Compile the Mojo Module (`lib_sigmoid`)

The Mojo code provides the core processing logic and kernels.

```bash
cd ~/code/darktable/mojo
pixi run mojo build -I . iop/sigmoid/lib.mojo --emit shared-lib -o libsigmoid_mojo.so
```

The resulting `libsigmoid_mojo.so` contains the following exported symbols used by the C bridge:
- `sigmoid_mojo_init`
- `sigmoid_mojo_destroy`
- `sigmoid_mojo_rgb_ratio`
- `sigmoid_mojo_per_channel`

## 2. Compile the C-Bridge Plugin (`sigmoid.c`)

The C part (`src/iop/sigmoid.c`) handles the darktable user interface and parameter management. It loads the Mojo shared library at runtime.

### Step-by-Step Build

**A. Generate Introspection Code:**
Use darktable's introspection tool to generate boilerplate for the plugin parameters.
```bash
perl tools/introspection/parser.pl src/ src/iop/sigmoid.c /tmp/introspection_sigmoid.c
```

**B. Compile and Link:**
Compile the generated code and link it into a shared library.

```bash
# Compilation
gcc -O3 -march=native -fPIC -fopenmp \
    $(pkg-config --cflags gtk+-3.0 glib-2.0 lcms2) \
    -Isrc -Isrc/iop -Ibuild/bin -include common/module_api.h -include iop/iop_api.h \
    -c /tmp/introspection_sigmoid.c -o /tmp/introspection_sigmoid.o

# Linking
gcc -shared -fPIC -fopenmp /tmp/introspection_sigmoid.o \
    -L/usr/lib/darktable -ldarktable -lm -lgomp \
    -o libsigmoid.so
```

## 3. Deployment

Both shared libraries must be placed in the darktable plugins directory so that darktable can find the plugin and the plugin can find the Mojo library.

```bash
# Copy the Mojo library
sudo cp mojo/libsigmoid_mojo.so /usr/lib/darktable/plugins/

# Copy the C bridge plugin
sudo cp libsigmoid.so /usr/lib/darktable/plugins/
```

> [!IMPORTANT]
> The C code in `src/iop/sigmoid.c` uses `dlopen("libsigmoid_mojo.so", RTLD_LAZY | RTLD_LOCAL)` in `init_global` to load the Mojo module. 
