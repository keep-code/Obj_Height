# 障碍物高度分类器 Makefile

CC = gcc
CFLAGS = -Wall -O2 -std=c99
LDFLAGS = -lm

TARGET = test_obstacle_height
SOURCES = obstacle_height_model.c test_obstacle_height.c
HEADERS = obstacle_height_model.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
