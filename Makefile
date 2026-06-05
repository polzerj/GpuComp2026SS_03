NVCC := nvcc
NVCC_FLAGS += -O3 --extended-lambda
COMPILER_OPTIONS += -Wall -Wextra
INCLUDES += -Iinclude

SRC_DIR := src
BUILD_DIR := build
EXECUTABLE := $(BUILD_DIR)/main

SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRC_FILES))

all: $(EXECUTABLE)

debug: NVCC_FLAGS += -DPREPROC_DEBUG -DPREPROC_TIMING
debug: $(EXECUTABLE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) --compiler-options "$(COMPILER_OPTIONS)" $(INCLUDES) -c $< -o $@ 

$(EXECUTABLE): $(OBJ_FILES) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) --compiler-options "$(COMPILER_OPTIONS)" $(OBJ_FILES) -o $@ -lnvjpeg

run: $(EXECUTABLE)
	@./$(EXECUTABLE)

clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: all clean run
