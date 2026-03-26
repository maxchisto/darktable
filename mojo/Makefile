MOJO = pixi run mojo
PLUGIN_DIR = /usr/lib/darktable/plugins

.PHONY: all build install bench clean

all: build

build: libsigmoid_mojo.so

libsigmoid_mojo.so: iop/sigmoid/lib.mojo iop/sigmoid/kernels.mojo
	$(MOJO) build -I . iop/sigmoid/lib.mojo --emit shared-lib -o libsigmoid_mojo.so

install: libsigmoid_mojo.so
	sudo cp libsigmoid_mojo.so $(PLUGIN_DIR)/

bench:
	$(MOJO) bench/sigmoid_bench.mojo

clean:
	rm -f libsigmoid_mojo.so
