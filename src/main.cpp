
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

#include <opencv2/opencv.hpp>
// #include <opencv2/cvconfig.h>
// #include <opencv2/core/types_c.h>

#include <openvr/openvr.h>

using namespace sl;
using namespace std;

// Zed Object
Camera zed;
InitParameters init_parameters;

//OpenGL 
GLuint shaderF;
GLuint program;
struct FramebufferDesc
{
    GLuint RenderTexture;
    GLuint Render_VRTexture;
    cudaGraphicsResource *pcuImageRes;
    Mat gpuImage; 
    GLuint shaderE;
    GLuint Framebuffer; 
};
FramebufferDesc leftEyeDesc;
FramebufferDesc rightEyeDesc;
cv::Mat convertColor; 
sl::Mat ColorGL; 
cv::cuda::GpuMat image_ocv_gpu; 
cudaGraphicsResource *Test;

// VR System
vr::IVRSystem *_pHMD;

// Functions 
void RenderStereoTargets(); 
void RenderScene(vr::Hmd_Eye nEye); 
bool CreateTexture(int nWidth, int nHeight, FramebufferDesc &framebufferDesc);
void ZedRetrieveImage();
void ViveDisplay();
void Close();
void intHandler(int) {
    Close();
}
void ZEDProjectionMatrix(FramebufferDesc &framebufferDesc); 
cv::Mat slMat2cvMat(Mat& input);
sl::Mat cvMat2slMat(cv::Mat& input); 
sl::Resolution res_; 
GLint m_nSceneMatrixLocation; 
// Create VBO for framebuffer rendering
GLuint texColorBuffer;
GLuint frameBuffer;
GLuint VAO; 

uint32_t m_nRenderWidth;
uint32_t m_nRenderHeight;
    float m_fNearClip = 0.1f;
 	float m_fFarClip = 30.0f;
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
    init_parameters.coordinate_system = COORDINATE_SYSTEM::LEFT_HANDED_Y_UP;
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
    // Shaders always begin with a version declaration, followed by a list of input and output variables, uniforms and its main function. Each shader's entry point is at its main function where we process any input variables and output the results in its output variables. Don't worry if you don't know what uniforms are, we'll get to those shortly. 
    {
        std::cout << "Compositor initialization failed. See log file for details" << endl;
    }
    else
    {
        cout << "COMPOSITOR OK" << endl;
    }

    _pHMD->GetRecommendedRenderTargetSize(&m_nRenderWidth, &m_nRenderHeight);
    vr::HmdMatrix44_t mat = _pHMD->GetProjectionMatrix( vr::Eye_Left, m_fNearClip, m_fFarClip );

    res_ = zed.getCameraInformation().camera_configuration.resolution;
    
    // CreateTexture(m_nRenderWidth, m_nRenderHeight, leftEyeDesc);
    // CreateTexture(m_nRenderWidth, m_nRenderHeight, rightEyeDesc);
    CreateTexture(res_.width, res_.height, leftEyeDesc);
    CreateTexture(res_.width, res_.height, rightEyeDesc);

    glGenFramebuffers(1, &frameBuffer);
    // At that point the framebuffer object is created, but is not complete yet
    // and can't be used.
    // For a framebuffer to be complete, there needs to be at least one buffer
    // attached (color, depth, stencil...), and at least one color attachement.
    // Besides, all attachement need to be complete (a texture attachement needs
    // to have its memory reserved...), and all attachements must have the same
    // number of multisamples.

    // Bind the framebuffer to work with it.
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);

    {  // You can test whether a framebuffer is complete as follows
        GLuint fb_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (fb_status != GL_FRAMEBUFFER_COMPLETE)
        {
        std::cout << "Framebuffer " << frameBuffer
                    << " is still incomplete and cannot yet be used." << std::endl;
        }
    }

    glGenTextures(1, &texColorBuffer);
    glBindTexture(GL_TEXTURE_2D, texColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res_.width, res_.height, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE,
               nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                            texColorBuffer, 0);
    {  // You can test whether a framebuffer is complete as follows
        GLuint fb_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (fb_status != GL_FRAMEBUFFER_COMPLETE)
        {
        std::cout << "Framebuffer " << frameBuffer
                    << " is still incomplete and cannot yet be used." << std::endl;
        }
        else
        {
        std::cout << "Framebuffer is now complete with one color attachement"
                    << std::endl;
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // Create the GLSL program that will run the fragment shader (defined at the top)
    // * Create the fragment shader from the string source
    // * Compile the shader and check for errors
    // * Create the GLSL program and attach the shader to it
    // * Link the program and check for errors
    // * Specify the uniform variable of the shader
    GLuint shaderF  = glCreateShader(GL_FRAGMENT_SHADER); //fragment shader
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
    m_nSceneMatrixLocation = glGetUniformLocation(program, "texImage"); 

    glUseProgram(program);

    // layout id of the position attribute in the vertex shader.
    const GLint colorAttrib = 0;

    /**
     * VERTEX ATTRIBUTE POINTERS
     **/
    // Enable the color attribute
    glEnableVertexAttribArray(colorAttrib);
    glVertexAttribPointer(colorAttrib, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                            (void *)(3 * sizeof(float)));

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
    glBindTexture(GL_TEXTURE_2D, 0);
    err = cudaGraphicsGLRegisterImage(&framebufferDesc.pcuImageRes, framebufferDesc.RenderTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // If any error are triggered, exit the program
    if (err != 0)
        return -1;

    return true;
}

void ZedRetrieveImage()
{
    vr::TrackedDevicePose_t m_rTrackedDevicePose[vr::k_unMaxTrackedDeviceCount];
    vr::HmdMatrix44_t mat = _pHMD->GetProjectionMatrix( vr::Eye_Left, m_fNearClip, m_fFarClip );
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
    cout << "RETRIEVE IMAGES OK" << endl;

    // Bind the default framebuffer (render on screen)
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
    // Clear the screen 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1, 0.5, 0, 1);
    glUseProgram(program); 
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, leftEyeDesc.RenderTexture);
    glBegin(GL_QUADS); 
    glTexCoord2f(0.0, 1.0);
    glVertex2f(-1.0, -1.0);
    glTexCoord2f(1.0, 1.0);
    glVertex2f(1.0, -1.0);
    glTexCoord2f(1.0, 0.0);
    glVertex2f(1.0, 1.0);
    glTexCoord2f(0.0, 0.0);
    glVertex2f(-1.0, 1.0);
    glEnd(); 

    // Draws the currently bound VAO using the currently bound shader program.
    // // This is drawn on the active framebuffer, which means that here we are
    // // doing rendering to texture.
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(program);
    // Bind the color texture of the framebuffer, where we have rendered the
    // scene
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texColorBuffer);
    // Dummy call to activate geometry shader: emit one point, and the geometry
    // shader will take care of emitting the required fullscreen quad.
     glBegin(GL_QUADS); 
    glTexCoord2f(0.0, 1.0);
    glVertex2f(-1.0, -1.0);
    glTexCoord2f(1.0, 1.0);
    glVertex2f(1.0, -1.0);
    glTexCoord2f(1.0, 0.0);
    glVertex2f(1.0, 1.0);
    glTexCoord2f(0.0, 0.0);
    glVertex2f(-1.0, 1.0);
    glEnd(); 


// glBindTexture(GL_TEXTURE_2D, rightEyeDesc.RenderTexture); 
//         glBegin(GL_QUADS);

//     glTexCoord2f(0.0, 1.0);
//     glVertex2f(0.0, -1.0);
//     glTexCoord2f(1.0, 1.0);
//     glVertex2f(1.0, -1.0);
//     glTexCoord2f(1.0, 0.0);
//     glVertex2f(1.0, 1.0);
//     glTexCoord2f(0.0, 0.0);
//     glVertex2f(0.0, 1.0);      
//       glEnd();


    if (_pHMD)
    {   
        // Render the final texture*
        // vr::IVRSystem::ComputeDistortion(); 
        vr::Texture_t leftEyeTexture = {(void *)(uintptr_t) texColorBuffer, vr::TextureType_OpenGL, vr::ColorSpace_Gamma};
        auto err = vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture);
        vr::Texture_t rightEyeTexture = {(void *)(uintptr_t)rightEyeDesc.RenderTexture, vr::TextureType_OpenGL, vr::ColorSpace_Gamma};
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


void Close()
{    
    if( _pHMD )
    {
    	vr::VR_Shutdown();
    	_pHMD = NULL;
    }   
    zed.close();
    glDeleteShader(shaderF);
    glDeleteShader(leftEyeDesc.shaderE); 
    glDeleteShader(rightEyeDesc.shaderE); 
    glDeleteProgram(program);
    glBindTexture(GL_TEXTURE_2D, 0);
    leftEyeDesc.gpuImage.free();
    rightEyeDesc.gpuImage.free();
}

void RenderStereoTargets()
{
	// glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );
	// glEnable( GL_MULTISAMPLE );

	// // Left Eye
 	// // RenderScene( vr::Eye_Left );	
    // // glGenFramebuffers(1, &leftEyeDesc.RenderTexture);
    // // glBindFramebuffer(GL_DRAW_FRAMEBUFFER, leftEyeDesc.RenderTexture );
    // glBindTexture(GL_TEXTURE_2D, leftEyeDesc.RenderTexture); 
 	// glViewport(0, 0, m_nRenderWidth, m_nRenderHeight );
    // glBlitFramebuffer( 0, 0, m_nRenderWidth, m_nRenderHeight, 0, 0, m_nRenderWidth, m_nRenderHeight, 
	// 	GL_COLOR_BUFFER_BIT,
 	// 	GL_LINEAR );
    // glBindTexture(GL_TEXTURE_2D, 0);
    // glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0 );	
    // glDisable( GL_MULTISAMPLE );

	// glEnable( GL_MULTISAMPLE );

	// // Right Eye
 	// // RenderScene( vr::Eye_Right ); 	
	
    // // glGenFramebuffers(1, &rightEyeDesc.RenderTexture);
    // // glBindFramebuffer(GL_DRAW_FRAMEBUFFER, rightEyeDesc.RenderTexture );
    // glBindTexture(GL_TEXTURE_2D, rightEyeDesc.RenderTexture); 
 	// glViewport(0, 0, m_nRenderWidth, m_nRenderHeight );
    // glBlitFramebuffer( 0, 0, m_nRenderWidth, m_nRenderHeight, 0, 0, m_nRenderWidth, m_nRenderHeight, 
	// 	GL_COLOR_BUFFER_BIT,
 	// 	GL_LINEAR  );
    // glBindTexture(GL_TEXTURE_2D, 0);
    // glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0 );
    // glDisable( GL_MULTISAMPLE );
}

void RenderScene(vr::Hmd_Eye nEye){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram( program );
    // glBindVertexArray( m_unSceneVAO );
    glBindTexture( GL_TEXTURE_2D, program );
    glBindVertexArray( 0 );

}

// Mapping between MAT_TYPE and CV_TYPE
int getOCVtype(sl::MAT_TYPE type) {
    int cv_type = -1;
    switch (type) {
        case MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    return cv_type;
}

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(sl::Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}

// Be careful about the memory owner, might want to use it like : 
//     cvMat2slMat(cv_mat).copyTo(sl_mat, sl::COPY_TYPE::CPU_CPU);
sl::Mat cvMat2slMat(cv::Mat& input) {
    sl::MAT_TYPE sl_type;
    switch (input.type()) {
        case CV_32FC1: sl_type = sl::MAT_TYPE::F32_C1;
            break;
        case CV_32FC2: sl_type = sl::MAT_TYPE::F32_C2;
            break;
        case CV_32FC3: sl_type = sl::MAT_TYPE::F32_C3;
            break;
        case CV_32FC4: sl_type = sl::MAT_TYPE::F32_C4;
            break;
        case CV_8UC1: sl_type = sl::MAT_TYPE::U8_C1;
            break;
        case CV_8UC2: sl_type = sl::MAT_TYPE::U8_C2;
            break;
        case CV_8UC3: sl_type = sl::MAT_TYPE::U8_C3;
            break;
        case CV_8UC4: sl_type = sl::MAT_TYPE::U8_C4;
            break;
        default: break;
    }
    return sl::Mat(input.cols, input.rows, sl_type, input.data, input.step, sl::MEM::CPU);
}
// void ZEDProjectionMatrix(){
    
//     //*** OPENVR
//     vr::HmdMatrix44_t prj[2] = { 
// 		_pHMD->GetProjectionMatrix(vr::Eye_Left, 0.01, 100),
// 		_pHMD->GetProjectionMatrix(vr::Eye_Right, 0.01, 100)}


//     //*** OPENGL 
//     glm::mat4 Model, View, Projection;
//     GLuint shaderProgram = glCreateShader(GL_SHADER); //fragment shader
//     GLint projection = glGetUniformLocation(shaderProgram, "Projection" );
//     glUniformMatrix4fv(projection, 1, GL_FALSE, glm::value_ptr(Projection));


//     glm::mat4 Proj; 
//     auto param_ = zed.getCameraInformation().camera_configuration.calibration_parameters; 
//     //LEFT
//     // Focal length of the left eye in pixels
//     float focal_left_x = param_.left_cam.fx;
//     float focal_left_y = parma_.left_cam.fy
//     // First radial distortion coefficient
//     float k1 = param_.left_cam.disto[0];
//     // Translation between left and right eye on z-axis
//     float tz = param_.T.z;
//     // Horizontal field of view of the left eye in degrees
//     float h_fov = param_.left_cam.h_fov;

//     //RIGHT
//     // Focal length of the left eye in pixels
//     float focal_right_x = param_.left_cam.fx;
//     // First radial distortion coefficient
//     float k1 = param_.left_cam.disto[0];
//     // Translation between left and right eye on z-axis
//     float tz = param_.T.z;
//     // Horizontal field of view of the left eye in degrees
//     float h_fov = param_.left_cam.h_fov;
// }
