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
find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)
set(OpenGL_GL_PREFERENCE LEGACY) 
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(SDL2 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS}
                    ${ZED_INCLUDE_DIRS}
                    ${SDL2_INCLUDE_DIRS}
                    ${GLEW_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR}
                 ${CUDA_LIBRARY_DIRS}
                 ${SDL2_LIBRARY_DIRS}
                 ${GLEW_LIBRARY_DIRS})

ADD_EXECUTABLE(${PROJECT_NAME} src/main.cpp )

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${ZED_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARIES} ${SDL2_LIBRARIES} PkgConfig::openvr)

