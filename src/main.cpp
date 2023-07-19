
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

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <eigen3/Eigen/Dense>

#include <openvr/openvr.h>

using namespace sl;
using namespace std;

// Zed Object
Camera zed;
InitParameters init_parameters;
vr::TrackedDevicePose_t m_rTrackedDevicePose[vr::k_unMaxTrackedDeviceCount];
glm::mat4 mvp_right; 
glm::mat4 mvp_left; 

//OpenGL 
GLuint shaderF;
GLuint shaderVertex; 
GLuint program;
GLuint programVertex;

struct FramebufferDesc
{
    GLuint RenderTexture;
    cudaGraphicsResource *pcuImageRes;
    Mat gpuImage; 
    GLuint Framebuffer; 
    GLuint texColorBuffer; 
    GLuint VBO, VAO, EBO;
};
FramebufferDesc leftEyeDesc;
FramebufferDesc rightEyeDesc;

// VR System
vr::IVRSystem *_pHMD;

// Functions 
void RenderStereoTargets(); 
void RenderScene(vr::Hmd_Eye nEye); 
bool CreateTexture(int nWidth, int nHeight, FramebufferDesc &framebufferDesc);
bool CreateShader(); 
bool CreateBuffer(FramebufferDesc &framebufferDesc); 
void ZedRetrieveImage();
glm::mat4 MVP(vr::TrackedDevicePose_t m_rTrackedDevicePose[64]); 
void ViveDisplay();
void Close();

void intHandler(int) {
    Close();
}
void ZEDProjectionMatrix(FramebufferDesc &framebufferDesc); 

sl::Resolution res_; 
GLint m_nSceneColor; 
GLint m_nSceneMatrixLocation; 

uint32_t m_nRenderWidth;
uint32_t m_nRenderHeight;
float m_fNearClip = 0.1f;
float m_fFarClip = 30.0f;

static const GLint ATTRIB_VERTICES_POS = 0;
static const GLint ATTRIB_COLOR_POS = 1;

// Simple fragment shader that switch red and blue channels (RGBA -->BGRA)
string strFragmentShad = ("uniform sampler2D texImage;\n"
        " void main() {\n"
        " vec4 color = texture2D(texImage, gl_TexCoord[0].st);\n"
        " gl_FragColor = vec4(color.b, color.g, color.r, color.a);\n}");

std::string vertexSource =
      "#version 440\n"
      "layout(location=0) in vec3 position;"
      "layout(location=1) in vec3 colors;"
      "layout(location=2) in vec2 texcoord;"
      "out vec2 Texcoord;"
      "layout(location=1) uniform mat4 trans;"

      "void main()"
      "{"
        "Texcoord = texcoord;"
        "gl_Position = trans * vec4(position, 1.0);"
      "}";
std::string fragmentSource =
      "#version 440\n"
      "in vec2 Texcoord;"
      "out vec4 outColor;"
      "layout(binding=0) uniform sampler2D imageTex;"
      "void main()"
      "{"
        " outColor = textureLod(imageTex, Texcoord, 0);"
        " outColor = vec4(outColor.b, outColor.g, outColor.r, outColor.a);"
      "}";


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
    init_parameters.coordinate_system = COORDINATE_SYSTEM::LEFT_HANDED_Y_UP;
    init_parameters.sdk_verbose=1;
    init_parameters.camera_fps = 30; 

    // Open the camera
    glutInit(&argc, argv);
    glutCreateWindow("GLEW Display");
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
    // Shaders always begin with a version declaration, followed by a list of input and output variables, uniforms and its main function. Each shader's entry point is at its main function where we process any input variables and output the results in its output variables. Don't worry if you don't know what uniforms are, we'll get to those shortly. 
    {
        std::cout << "Compositor initialization failed. See log file for details" << endl;
    }
    else
    {
        cout << "COMPOSITOR OK" << endl;
    }

    _pHMD->GetRecommendedRenderTargetSize(&m_nRenderWidth, &m_nRenderHeight); //VR resolution 1440 x 1600
    // vr::HmdMatrix44_t mat = _pHMD->GetProjectionMatrix( vr::Eye_Left, m_fNearClip, m_fFarClip );
    res_ = zed.getCameraInformation().camera_configuration.resolution;

    CreateShader(); 
    CreateBuffer(leftEyeDesc); 
    CreateBuffer(rightEyeDesc); 
    CreateTexture(m_nRenderWidth, m_nRenderHeight, leftEyeDesc);
    CreateTexture(m_nRenderWidth, m_nRenderHeight, rightEyeDesc);
    cout << "Width HMD VIVE: " << m_nRenderWidth << endl; 
    cout << "Height HMD VIVE: " << m_nRenderHeight << endl; 
    cout << "Width ZED Mini: " << res_.width << endl; 
    cout << "Height ZED Mini: " << res_.height << endl; 

    glutDisplayFunc(ZedRetrieveImage);
    glutCloseFunc(Close);
    glutMainLoop();

    return EXIT_SUCCESS;
}

bool CreateTexture(int nWidth, int nHeight, FramebufferDesc &framebufferDesc)
{
    // Create glTexture 
    cudaError_t err;
    glGenTextures(1, &framebufferDesc.RenderTexture);
    glBindTexture(GL_TEXTURE_2D, framebufferDesc.RenderTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, nWidth, nHeight, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    err = cudaGraphicsGLRegisterImage(&framebufferDesc.pcuImageRes, framebufferDesc.RenderTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    
    // Create glBuffer 
    glGenFramebuffers(1, &framebufferDesc.Framebuffer);
    // At that point the framebuffer object is created, but is not complete yet
    // and can't be used.
    // For a framebuffer to be complete, there needs to be at least one buffer
    // attached (color, depth, stencil...), and at least one color attachement.
    // Besides, all attachement need to be complete (a texture attachement needs
    // to have its memory reserved...), and all attachements must have the same
    // number of multisamples.

    // Bind the framebuffer to work with it.
    glBindFramebuffer(GL_FRAMEBUFFER, framebufferDesc.Framebuffer);

    {  // You can test whether a framebuffer is complete as follows
        GLuint fb_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (fb_status != GL_FRAMEBUFFER_COMPLETE)
        {
        std::cout << "Framebuffer " << framebufferDesc.Framebuffer
                    << " is still incomplete and cannot yet be used." << std::endl;
        }
    }

    glGenTextures(1, &framebufferDesc.texColorBuffer);
    glBindTexture(GL_TEXTURE_2D, framebufferDesc.texColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res_.width, res_.height, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE,
               nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                            framebufferDesc.texColorBuffer, 0);
    {  // You can test whether a framebuffer is complete as follows
        GLuint fb_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (fb_status != GL_FRAMEBUFFER_COMPLETE)
        {
        std::cout << "Framebuffer " << framebufferDesc.Framebuffer
                    << " is still incomplete and cannot yet be used." << std::endl;
        }
        else
        {
        std::cout << "Framebuffer is now complete with one color attachement"
                    << std::endl;
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // If any error are triggered, exit the program
    if (err != 0)
        return -1;

    return true;
}

void ZedRetrieveImage()
{
    vr::HmdMatrix44_t mat = _pHMD->GetProjectionMatrix( vr::Eye_Left, m_fNearClip, m_fFarClip );
    vr::VRCompositor()->WaitGetPoses(m_rTrackedDevicePose, vr::k_unMaxTrackedDeviceCount, NULL, 0);
    const GLint uniTrans = 1;
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
            cudaGraphicsMapResources(1, &leftEyeDesc.pcuImageRes, 0);
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

    //LEFT EYE 
    glm::mat4 model_left = glm::mat4(1.0f);
    glm::mat4 scaling_left = glm::scale(model_left, glm::vec3(1.0,1.0,1.0));  
    // glm::mat4 scaling_left = glm::scale(model_left, glm::vec3(0.603,0.603,0.603));  
    // mvp_left = glm::translate(scaling_left, glm::vec3(0.426, -0.643, 0.0));	// move parametre Carole 
    // mvp_left = glm::translate(scaling_left, glm::vec3(0.5, -0.75, 0.0));	// move parametres ZED mini
    mvp_left = glm::translate(scaling_left, glm::vec3(0.6, -0.75, 0.0));	// move parametres ZED 2
    // mvp_left = glm::translate(scaling_left, glm::vec3(0.428, -0.643, 0.0));	// move parametre Paul Audoyer 
    // Bind the default framebuffer (render on screen)
    glBindFramebuffer(GL_FRAMEBUFFER, leftEyeDesc.Framebuffer);
    // Clear the screen 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1, 0.4, 0, 1);
    glUseProgram(program); 
    glBindVertexArray(leftEyeDesc.VAO); 
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, leftEyeDesc.RenderTexture);
    glUniformMatrix4fv(uniTrans, 1, GL_FALSE, glm::value_ptr(mvp_left));
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, leftEyeDesc.texColorBuffer);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);

    //RIGHT EYE 
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 scaling = glm::scale(model, glm::vec3(1.0,1.0,1.0));  
    // glm::mat4 scaling = glm::scale(model, glm::vec3(0.603,0.603,0.603));  
    // mvp_right = glm::translate(scaling, glm::vec3(-0.00048, -0.643, 0.0f));	// move ZED = glm::vec3(-0.15f, -0.5f, 0.0f) || ZED2 = glm::vec3(-0.36f, -0.5f, 0.0f)
    // mvp_right = glm::translate(scaling, glm::vec3(-0.000, -0.643, 0.0f));	// move ZED = glm::vec3(-0.15f, -0.5f, 0.0f) || ZED2 = glm::vec3(-0.36f, -0.5f, 0.0f)
    // mvp_right = glm::translate(scaling, glm::vec3(0.5, -0.75, 0.0f));	// move ZED parametres ZED mini
    mvp_right = glm::translate(scaling, glm::vec3(0.4, -0.75, 0.0f));	// move ZED parametres ZED 2
    // Bind the default framebuffer (render on screen)
    glBindFramebuffer(GL_FRAMEBUFFER, rightEyeDesc.Framebuffer);
    // Clear the screen 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0, 0, 0, 1);
    glUseProgram(program); 
    glBindVertexArray(rightEyeDesc.VAO); 
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, rightEyeDesc.RenderTexture);
    glUniformMatrix4fv(uniTrans, 1, GL_FALSE, glm::value_ptr(mvp_right));
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, rightEyeDesc.texColorBuffer);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);


    if (_pHMD)
    {   
        // Render the final texture
        vr::Texture_t leftEyeTexture = {(void *)(uintptr_t) leftEyeDesc.texColorBuffer, vr::TextureType_OpenGL, vr::ColorSpace_Gamma};
        auto err = vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture);
        vr::Texture_t rightEyeTexture = {(void *)(uintptr_t)rightEyeDesc.texColorBuffer, vr::TextureType_OpenGL, vr::ColorSpace_Gamma};
        vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture);
    
        // We want to make sure the glFinish waits for the entire present to complete, not just the submission
        // of the command. So, we do a clear here right here so the glFinish will wait fully for the swap.    
        glutSwapBuffers();
        glFlush();
        glFinish();  
        glutPostRedisplay();
   
    }
 
    else
    {
        std::cout << "HMD not detected";
    }
}

bool CreateShader()
{
    cout << "CreateProgram" << endl; 
    GLuint vertex, fragment; 
    // vertex shader
    const char *VERTEX_SHADER = vertexSource.c_str();
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &VERTEX_SHADER, NULL);
    glCompileShader(vertex);
    // fragment Shader
    const char *FRAGMENT_SHADER = fragmentSource.c_str();
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &FRAGMENT_SHADER, NULL);
    glCompileShader(fragment);
    // shader Program
    program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glBindFragDataLocation(program, 0, "outColor");
    glUniform1i(glGetUniformLocation(program, "texImage"), 0);
    glLinkProgram(program);
    glUseProgram(program); 

    return true;

}

bool CreateBuffer(FramebufferDesc &framebufferDesc)
{
    float vertices[] = {
    // positions          // colors           // texture coords
    1.0f,  -1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
    1.0f, 1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
    -1.0f, 1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
    -1.0f,  -1.0f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {  
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };

    glGenVertexArrays(1, &framebufferDesc.VAO);
    glGenBuffers(1, &framebufferDesc.VBO);
    glGenBuffers(1, &framebufferDesc.EBO);

    glBindVertexArray(framebufferDesc.VAO);

    glBindBuffer(GL_ARRAY_BUFFER, framebufferDesc.VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, framebufferDesc.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    return true; 
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
    glDeleteVertexArrays(1, &leftEyeDesc.VAO);
    glDeleteBuffers(1, &leftEyeDesc.VBO);
    glDeleteBuffers(1, &leftEyeDesc.EBO);
    glDeleteVertexArrays(1, &rightEyeDesc.VAO);
    glDeleteBuffers(1, &rightEyeDesc.VBO);
    glDeleteBuffers(1, &rightEyeDesc.EBO);
    glBindTexture(GL_TEXTURE_2D, 0);
    leftEyeDesc.gpuImage.free();
    rightEyeDesc.gpuImage.free();
}

