# Build bang MinGW (gcc). Chay: mingw32-make  (hoac make)
# Duong dan tuyet doi den thu vien neu dich vu o khac thi sua 2 dong duoi.
MUJOCO_DIR = ../../MuJoCo
GLFW_DIR   = ../../third_party/glfw-3.4.bin.WIN64

PROJECT_INC = src/logic/include

COMPILER = gcc
TARGET   = bin/belt.exe

SRC = src/main.c src/logic/logic.c

CFLAGS  = -O2 -I $(PROJECT_INC) -I $(MUJOCO_DIR)/include -I $(GLFW_DIR)/include
LDFLAGS = -L $(MUJOCO_DIR)/lib -L $(GLFW_DIR)/lib-mingw-w64 \
          -lmujoco -lglfw3 -lgdi32 -lshell32 -lopengl32

all: $(TARGET)

$(TARGET): $(SRC) | bin
	$(COMPILER) $(CFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)
	copy /Y "$(MUJOCO_DIR)\bin\mujoco.dll" bin\ >nul
	copy /Y "$(GLFW_DIR)\lib-mingw-w64\glfw3.dll" bin\ >nul

bin:
	if not exist bin mkdir bin

clean:
	rmdir /S /Q bin

.PHONY: all bin clean run

run: $(TARGET)
	 cd bin && belt.exe
