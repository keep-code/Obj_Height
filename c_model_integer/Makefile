# 障碍物高度分类器 Makefile (整型版本)

CC = gcc
CFLAGS = -Wall -O2 -std=c99
LDFLAGS = -lm

TARGET = test_obstacle_height_int
SOURCES = obstacle_height_model_int.c test_obstacle_height_int.c
HEADERS = obstacle_height_model_int.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
