# Build bang MinGW (gcc). Chay: mingw32-make  (hoac make)
# Duong dan tuyet doi den thu vien neu dich vu o khac thi sua 2 dong duoi.
MUJOCO_DIR = ../../MuJoCo
GLFW_DIR   = ../../third_party/glfw-3.4.bin.WIN64

COMPILER = gcc
TARGET   = belt.exe

CFLAGS  = -O2 -I $(MUJOCO_DIR)/include -I $(GLFW_DIR)/include
LDFLAGS = -L $(MUJOCO_DIR)/lib -L $(GLFW_DIR)/lib-mingw-w64 \
          -lmujoco -lglfw3 -lgdi32 -lshell32 -lopengl32

all: $(TARGET)

$(TARGET): main.c
	$(COMPILER) $(CFLAGS) main.c $(LDFLAGS) -o $(TARGET)
	copy /Y "$(MUJOCO_DIR)\bin\mujoco.dll" . >nul
	copy /Y "$(GLFW_DIR)\lib-mingw-w64\glfw3.dll" . >nul

run: $(TARGET)
	$(TARGET)

clean:
	del /Q $(TARGET)

.PHONY: all run clean

