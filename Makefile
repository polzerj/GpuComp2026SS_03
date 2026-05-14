NVCC := nvcc
NVCC_FLAGS += -O3 -Wall -Wextra
INCLUDES += -Iinclude

SRC_DIR := src
BUILD_DIR := build
EXECUTABLE := $(BUILD_DIR)/main

SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRC_FILES))

all: $(EXECUTABLE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(EXECUTABLE): $(OBJ_FILES) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(OBJ_FILES) -o $@

run:
	./$(EXECUTABLE)

clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: all clean run
