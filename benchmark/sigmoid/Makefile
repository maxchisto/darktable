CC=clang
CFLAGS=-O3 -Wall
LDFLAGS=-lOpenCL -lm

TARGET=benchmark_sigmoid
SRC=benchmark_sigmoid.c

all: $(TARGET)
	
.PHONY: force
$(TARGET): $(SRC) force
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: all
	./$(TARGET)
