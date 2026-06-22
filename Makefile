NVCC := nvcc
GPU_ARCH := sm_75
NVCC_FLAGS += -O3 \
              -arch=$(GPU_ARCH) \
              --use_fast_math \
              --extended-lambda \
              -Xptxas -O3 \
              -Xptxas -dlcm=ca \
              -Xptxas -v \
              #-DENABLE_TIMING #-DDRAW_CORNERS -DDRAW_COLORED_EDGES
COMPILER_OPTIONS += -Wall -Wextra 
INCLUDES += -Iinclude 
LIBS += -lnvjpeg -lcudadevrt

SRC_DIR := src
BUILD_DIR := build
EXECUTABLE := $(BUILD_DIR)/main
IMAGES_DIR ?= ../data/

SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRC_FILES))

all: $(EXECUTABLE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) --compiler-options "$(COMPILER_OPTIONS)" $(INCLUDES) -c $< -o $@

$(EXECUTABLE): $(OBJ_FILES) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) --compiler-options "$(COMPILER_OPTIONS)" $(OBJ_FILES) -o $@ $(LIBS)

run: $(EXECUTABLE)
	@./$(EXECUTABLE) $(IMG)

run_colored: $(EXECUTABLE)
	@./$(EXECUTABLE) $(IMAGES_DIR)/input-example-tuwel.jpg

run_all: $(EXECUTABLE)
	@./$(EXECUTABLE) $(IMAGES_DIR)/*.jpg

clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: all clean run
