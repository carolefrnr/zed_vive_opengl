# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/carole/src/zed_display_opengl

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/carole/src/zed_display_opengl/build

# Include any dependencies generated for this target.
include CMakeFiles/ZED_Vive_Display.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ZED_Vive_Display.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ZED_Vive_Display.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ZED_Vive_Display.dir/flags.make

CMakeFiles/ZED_Vive_Display.dir/src/main.cpp.o: CMakeFiles/ZED_Vive_Display.dir/flags.make
CMakeFiles/ZED_Vive_Display.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/ZED_Vive_Display.dir/src/main.cpp.o: CMakeFiles/ZED_Vive_Display.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/carole/src/zed_display_opengl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ZED_Vive_Display.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ZED_Vive_Display.dir/src/main.cpp.o -MF CMakeFiles/ZED_Vive_Display.dir/src/main.cpp.o.d -o CMakeFiles/ZED_Vive_Display.dir/src/main.cpp.o -c /home/carole/src/zed_display_opengl/src/main.cpp

CMakeFiles/ZED_Vive_Display.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ZED_Vive_Display.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/carole/src/zed_display_opengl/src/main.cpp > CMakeFiles/ZED_Vive_Display.dir/src/main.cpp.i

CMakeFiles/ZED_Vive_Display.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ZED_Vive_Display.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/carole/src/zed_display_opengl/src/main.cpp -o CMakeFiles/ZED_Vive_Display.dir/src/main.cpp.s

# Object files for target ZED_Vive_Display
ZED_Vive_Display_OBJECTS = \
"CMakeFiles/ZED_Vive_Display.dir/src/main.cpp.o"

# External object files for target ZED_Vive_Display
ZED_Vive_Display_EXTERNAL_OBJECTS =

ZED_Vive_Display: CMakeFiles/ZED_Vive_Display.dir/src/main.cpp.o
ZED_Vive_Display: CMakeFiles/ZED_Vive_Display.dir/build.make
ZED_Vive_Display: /usr/local/zed/lib/libsl_zed.so
ZED_Vive_Display: /usr/lib/x86_64-linux-gnu/libopenblas.so
ZED_Vive_Display: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
ZED_Vive_Display: /usr/lib/x86_64-linux-gnu/libnvidia-encode.so
ZED_Vive_Display: /usr/lib/x86_64-linux-gnu/libcuda.so
ZED_Vive_Display: /usr/local/cuda/lib64/libcudart.so
ZED_Vive_Display: /usr/lib/x86_64-linux-gnu/libGL.so
ZED_Vive_Display: /usr/lib/x86_64-linux-gnu/libGLU.so
ZED_Vive_Display: /usr/lib/x86_64-linux-gnu/libglut.so
ZED_Vive_Display: /usr/lib/x86_64-linux-gnu/libXmu.so
ZED_Vive_Display: /usr/lib/x86_64-linux-gnu/libXi.so
ZED_Vive_Display: /usr/lib/x86_64-linux-gnu/libGLEW.so
ZED_Vive_Display: /usr/lib/x86_64-linux-gnu/libopenvr_api.so
ZED_Vive_Display: /usr/lib/x86_64-linux-gnu/libdl.so
ZED_Vive_Display: CMakeFiles/ZED_Vive_Display.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/carole/src/zed_display_opengl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ZED_Vive_Display"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ZED_Vive_Display.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ZED_Vive_Display.dir/build: ZED_Vive_Display
.PHONY : CMakeFiles/ZED_Vive_Display.dir/build

CMakeFiles/ZED_Vive_Display.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ZED_Vive_Display.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ZED_Vive_Display.dir/clean

CMakeFiles/ZED_Vive_Display.dir/depend:
	cd /home/carole/src/zed_display_opengl/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/carole/src/zed_display_opengl /home/carole/src/zed_display_opengl /home/carole/src/zed_display_opengl/build /home/carole/src/zed_display_opengl/build /home/carole/src/zed_display_opengl/build/CMakeFiles/ZED_Vive_Display.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ZED_Vive_Display.dir/depend

