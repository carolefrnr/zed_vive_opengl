
///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2020, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/***********************************************************************************************
 ** This sample demonstrates how to grab images and depth map with the ZED SDK                **
 ** The GPU buffer is ingested directly into OpenGL texture to avoid GPU->CPU readback time   **
 ** For the Left image, a GLSL shader is used for RGBA-->BGRA transformation, as an example   **
 ***********************************************************************************************/

#include <stdio.h>
#include <string.h>
#include <ctime>
#include <signal.h>

#include <sl/Camera.hpp>
#include <thread>

#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <openvr/openvr.h>

using namespace sl;
using namespace std;

// Zed Object
Camera zed;
InitParameters init_parameters;
GLuint shaderF;
GLuint program;
// VR System
vr::IVRSystem *_pHMD;
struct FramebufferDesc
{
    GLuint RenderTexture;
    cudaGraphicsResource *pcuImageRes;
    Mat gpuImage;
};
FramebufferDesc leftEyeDesc;
FramebufferDesc rightEyeDesc;
bool CreateTexture(int nWidth, int nHeight, FramebufferDesc &framebufferDesc);
void ZedRetrieveImage();
void ViveDisplay();
void Close();
void intHandler(int) {
    Close();
}
uint32_t m_nRenderWidth;
uint32_t m_nRenderHeight;

// Simple fragment shader that switch red and blue channels (RGBA -->BGRA)
string strFragmentShad = ("uniform sampler2D texImage;\n"
        " void main() {\n"
        " vec4 color = texture2D(texImage, gl_TexCoord[0].st);\n"
        " gl_FragColor = vec4(color.b, color.g, color.r, color.a);\n}");

// Main loop for acquisition and rendering :
// * grab from the ZED SDK
// * Map cuda and opengl resources and copy the GPU buffer into a CUDA array
// * Use the OpenGL texture to render on the screen

int main(int argc, char **argv)
{
    signal(SIGINT, intHandler);
    // Create ZED objects
    init_parameters.camera_resolution = RESOLUTION::HD1080;
    init_parameters.depth_mode = DEPTH_MODE::NONE;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    init_parameters.sdk_verbose=1;

    // Open the camera
    glutInit(&argc, argv);
    glutCreateWindow("GLEW Test");
    GLenum err = glewInit();
    cout << "OPENGL OK" << endl;
    // Init GLEW Library
    glewInit();
    // Setup our ZED Camera (construct and Init)
    if (argc == 2) // Use in SVO playback mode
        init_parameters.input.setFromSVOFile(String(argv[1]));

    init_parameters.depth_mode = DEPTH_MODE::PERFORMANCE;
    ERROR_CODE err1 = zed.open(init_parameters);

    // ERRCODE display
    if (err1 != ERROR_CODE::SUCCESS)
    {
        cout << "ZED Opening Error: " << err1 << endl;
        zed.close();
        return EXIT_FAILURE;
    }
    cout << "ZED OK" << endl;

    // Init VR
    vr::EVRInitError eError = vr::VRInitError_None;
    _pHMD = vr::VR_Init(&eError, vr::VRApplication_Scene);

    if (eError != vr::VRInitError_None)
    {
        _pHMD = NULL;
        std::cout << "Unable to init VR runtime" << endl;
    }
    else
    {
        cout << "HMD OK" << endl;
    }

    vr::EVRInitError peError = vr::VRInitError_None;
    if (!vr::VRCompositor())
    {
        std::cout << "Compositor initialization failed. See log file for details" << endl;
    }
    else
    {
        cout << "COMPOSITOR OK" << endl;
    }


    _pHMD->GetRecommendedRenderTargetSize(&m_nRenderWidth, &m_nRenderWidth);
    auto res_ = zed.getCameraInformation().camera_configuration.resolution;
    auto param_ = zed.getCameraInformation().camera_configuration.calibration_parameters; 

    CreateTexture(res_.width, res_.height, leftEyeDesc);
    CreateTexture(res_.width, res_.height, rightEyeDesc);

    glutDisplayFunc(ZedRetrieveImage);
    glutCloseFunc(Close);
    glutMainLoop();

    return EXIT_SUCCESS;
}

bool CreateTexture(int nWidth, int nHeight, FramebufferDesc &framebufferDesc)
{
    cudaError_t err;
    glActiveTexture(GL_TEXTURE0); 
    glGenTextures(1, &framebufferDesc.RenderTexture);
    glBindTexture(GL_TEXTURE_2D, framebufferDesc.RenderTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, nWidth, nHeight, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, nWidth, nHeight, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    err = cudaGraphicsGLRegisterImage(&framebufferDesc.pcuImageRes, framebufferDesc.RenderTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    // If any error are triggered, exit the program
    if (err != 0)
        return -1;

    // Create the GLSL program that will run the fragment shader (defined at the top)
    // * Create the fragment shader from the string source
    // * Compile the shader and check for errors
    // * Create the GLSL program and attach the shader to it
    // * Link the program and check for errors
    // * Specify the uniform variable of the shader
    GLuint shaderF = glCreateShader(GL_FRAGMENT_SHADER); //fragment shader
    const char* pszConstString = strFragmentShad.c_str();
    glShaderSource(shaderF, 1, (const char**) &pszConstString, NULL);

    // Compile the shader source code and check
    glCompileShader(shaderF);
    GLint compile_status = GL_FALSE;
    glGetShaderiv(shaderF, GL_COMPILE_STATUS, &compile_status);
    if (compile_status != GL_TRUE) return -2;

    // Create the progam for both V and F Shader
    program = glCreateProgram();
    glAttachShader(program, shaderF);

    // Link the program and check for errors
    glLinkProgram(program);
    GLint link_status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);
    if (link_status != GL_TRUE) return -2;

    // Set the uniform variable for texImage (sampler2D) to the texture unit (GL_TEXTURE0 by default --> id = 0)
    glUniform1i(glGetUniformLocation(program, "texImage"), 0);
    glUseProgram(program); 
    return true;
}

void ZedRetrieveImage()
{
    vr::TrackedDevicePose_t m_rTrackedDevicePose[vr::k_unMaxTrackedDeviceCount];
    vr::VRCompositor()->WaitGetPoses(m_rTrackedDevicePose, vr::k_unMaxTrackedDeviceCount, NULL, 0);
    if (zed.grab() == ERROR_CODE::SUCCESS)
    {
        // Map GPU Resource for left image
        // With OpenGL textures, we need to use the cudaGraphicsSubResourceGetMappedArray CUDA functions. It will link/sync the OpenGL texture with a CUDA cuArray
        // Then, we just have to copy our GPU Buffer to the CudaArray (DeviceToDevice copy) and the texture will contain the GPU buffer content.
        // That's the most efficient way since we don't have to go back on the CPU to render the texture. Make sure that retrieveXXX() functions of the ZED SDK
        // are used with sl::MEM::GPU parameters.
        if (zed.retrieveImage(leftEyeDesc.gpuImage, VIEW::LEFT, MEM::GPU) == ERROR_CODE::SUCCESS)
        {
            cudaArray_t ArrImLeft;
            cudaGraphicsMapResources(1, &leftEyeDesc.gpuImage, 0);
            cudaGraphicsSubResourceGetMappedArray(&ArrImLeft, leftEyeDesc.pcuImageRes, 0, 0);
            cudaMemcpy2DToArray(ArrImLeft, 0, 0, leftEyeDesc.gpuImage.getPtr<sl::uchar1>(MEM::GPU), leftEyeDesc.gpuImage.getStepBytes(MEM::GPU), leftEyeDesc.gpuImage.getWidth() * sizeof(sl::uchar4), leftEyeDesc.gpuImage.getHeight(), cudaMemcpyDeviceToDevice);
            cudaGraphicsUnmapResources(1, &leftEyeDesc.pcuImageRes, 0);
        }
        if (zed.retrieveImage(rightEyeDesc.gpuImage, VIEW::RIGHT, MEM::GPU) == ERROR_CODE::SUCCESS)
        {
            cudaArray_t ArrImRight;
            cudaGraphicsMapResources(1, &rightEyeDesc.pcuImageRes, 0);
            cudaGraphicsSubResourceGetMappedArray(&ArrImRight, rightEyeDesc.pcuImageRes, 0, 0);
            cudaMemcpy2DToArray(ArrImRight, 0, 0, rightEyeDesc.gpuImage.getPtr<sl::uchar1>(MEM::GPU), rightEyeDesc.gpuImage.getStepBytes(MEM::GPU), rightEyeDesc.gpuImage.getWidth() * sizeof(sl::uchar4), rightEyeDesc.gpuImage.getHeight(), cudaMemcpyDeviceToDevice);
            cudaGraphicsUnmapResources(1, &rightEyeDesc.pcuImageRes, 0);
        }
    }
    else
    {
        cout << "ZED cannot retrieve images" << endl;
    }

    if (_pHMD)
    {
        vr::Texture_t leftEyeTexture = {(void *)(uintptr_t)leftEyeDesc.RenderTexture, vr::TextureType_OpenGL, vr::ColorSpace_Gamma};
        auto err = vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture);
        // cout << err;
        vr::Texture_t rightEyeTexture = {(void *)(uintptr_t)rightEyeDesc.RenderTexture, vr::TextureType_OpenGL, vr::ColorSpace_Gamma};
        vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture);

        // We want to make sure the glFinish waits for the entire present to complete, not just the submission
        // of the command. So, we do a clear here right here so the glFinish will wait fully for the swap.
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1, 0.5, 0, 1);
        glutSwapBuffers();
        glFlush();
        glFinish();  
        glutPostRedisplay();
    }
            // Render the final texture
 
    else
    {
        std::cout << "HMD not detected";
    }
}


void Close()
{    
    if( _pHMD )
    {
    	vr::VR_Shutdown();
    	_pHMD = NULL;
    }   
    zed.close();
    glDeleteShader(shaderF);
    glDeleteProgram(program);
    glBindTexture(GL_TEXTURE_2D, 0);
    leftEyeDesc.gpuImage.free();
    rightEyeDesc.gpuImage.free();
}
