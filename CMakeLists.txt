CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(ZED_Vive_Display)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE "Release")

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

include(FindPkgConfig)
pkg_check_modules(openvr REQUIRED IMPORTED_TARGET openvr)
pkg_check_modules(OGRE REQUIRED IMPORTED_TARGET OGRE)
find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)
find_package(OpenCV REQUIRED)
set(OpenGL_GL_PREFERENCE LEGACY) 
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(SDL2 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(${OGRE_INCLUDE_DIRS})
include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})
link_directories(${SDL2_LIBRARY_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${OGRE_LIBRARY_DIRS})

ADD_EXECUTABLE(${PROJECT_NAME} src/main.cpp )

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${ZED_LIBS} ${OpenCV_LIBRARIES} ${OPENGL_LIBRARIES} ${OGRE_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARIES} ${SDL2_LIBRARIES} PkgConfig::openvr)

