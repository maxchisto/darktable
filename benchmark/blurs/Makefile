CC=gcc
CFLAGS=-O3 -Wall
LDFLAGS=-lOpenCL -lm

TARGET=benchmark_blurs
SRC=benchmark_blurs.c

all: $(TARGET)
	
.PHONY: force
$(TARGET): $(SRC) force
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: all
	./$(TARGET)
