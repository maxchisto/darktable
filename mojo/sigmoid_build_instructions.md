# Build Instructions

## 1. Compile the Mojo Module (`libsigmoid_mojo.so`)

The Mojo code provides the core processing logic and kernels.

```bash
cd ~/code/darktable/mojo
make build 
```

The resulting `libsigmoid_mojo.so` contains the following exported symbols used by the C bridge:
- `sigmoid_mojo_init`
- `sigmoid_mojo_destroy`
- `sigmoid_mojo_rgb_ratio`
- `sigmoid_mojo_per_channel`

## 2. Compile the C-Bridge Plugin (`libsigmoid.so`)

The C part (`src/iop/sigmoid.c`) handles the darktable user interface and parameter management. It loads the Mojo shared library at runtime.

```bash
./build_sigmoid_iop.sh
```


## 3. Installation

Both shared libraries must be placed in the darktable plugins directory so that darktable can find the plugin and the plugin can find the Mojo library.

```bash
# Copy the Mojo library
sudo cp libsigmoid_mojo.so /usr/lib/darktable/plugins/

# Copy the C bridge plugin
sudo cp libsigmoid.so /usr/lib/darktable/plugins/
```

> The C code in `src/iop/sigmoid.c` uses `dlopen("libsigmoid_mojo.so", RTLD_LAZY | RTLD_LOCAL)` in `init_global` to load the Mojo module. 
