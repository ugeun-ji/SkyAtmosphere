/*
 * Copyright ?2012-2015 Graham Sellers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "../include/sb7.h"
#include "../include/sb7ktx.h"
#include "../include/shader.h"
#include "../include/vmath.h"

class SkyAtmosphere : public sb7::application
{
    void init()
    {
        static const char title[] = "Sky Atmosphere";

        sb7::application::init();

        memcpy(info.title, title, sizeof(title));
    }

    virtual void startup()
    {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        // Load shaders.
        GLuint fullscreenQuadVertexShader = sb7::shader::load("resources/shaders/fullscreen_quad.vs.glsl", GL_VERTEX_SHADER);

        // -----------------------------------------------------------------
        // Atmospheric Parameters Uniform Buffer.
        // -----------------------------------------------------------------
        glGenBuffers(1, &atmosphericParametersUniformBuffer);
        glBindBuffer(GL_UNIFORM_BUFFER, atmosphericParametersUniformBuffer);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(AtmosphericParametersUniformBuffer), NULL, GL_STATIC_DRAW);

        // -----------------------------------------------------------------
        // Transmittance LUT.
        // -----------------------------------------------------------------
        GLuint transmittanceLutPixelShader = sb7::shader::load("resources/shaders/transmittance_lut.fs.glsl", GL_FRAGMENT_SHADER);

        transmittanceLutProgram = glCreateProgram();

        glAttachShader(transmittanceLutProgram, fullscreenQuadVertexShader);
        glAttachShader(transmittanceLutProgram, transmittanceLutPixelShader);

        glLinkProgram(transmittanceLutProgram);

        uniforms.transmittanceLUT.width = glGetUniformLocation(transmittanceLutProgram, "width");
        uniforms.transmittanceLUT.height = glGetUniformLocation(transmittanceLutProgram, "height");

        glGenTextures(1, &transmittanceLutTexture);
        glBindTexture(GL_TEXTURE_2D, transmittanceLutTexture);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, 256, 64);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // -----------------------------------------------------------------
        // MultiScattering LUT.
        // -----------------------------------------------------------------
        GLuint multiScatteringLutComputeShader = sb7::shader::load("resources/shaders/multiscattering_lut.cs.glsl", GL_COMPUTE_SHADER);

        multiScatteringLutProgram = glCreateProgram();

        glAttachShader(multiScatteringLutProgram, multiScatteringLutComputeShader);

        glLinkProgram(multiScatteringLutProgram);

        uniforms.multiScatteringLUT.transmittanceLutTexture = glGetUniformLocation(multiScatteringLutProgram, "transmittanceLutTexture");
        uniforms.multiScatteringLUT.outputTexture = glGetUniformLocation(multiScatteringLutProgram, "outputTexture");

        glGenTextures(1, &multiScatteringLutTexture);
        glBindTexture(GL_TEXTURE_2D, multiScatteringLutTexture);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, 32, 32);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // -----------------------------------------------------------------
        // SkyView LUT.
        // -----------------------------------------------------------------
        GLuint skyViewLutPixelShader = sb7::shader::load("resources/shaders/skyview_lut.fs.glsl", GL_FRAGMENT_SHADER);

        skyViewLutProgram = glCreateProgram();

        glAttachShader(skyViewLutProgram, fullscreenQuadVertexShader);
        glAttachShader(skyViewLutProgram, skyViewLutPixelShader);

        glLinkProgram(skyViewLutProgram);

        uniforms.skyviewLUT.transmittanceLutTexture = glGetUniformLocation(skyViewLutProgram, "transmittanceLutTexture");

        glGenTextures(1, &skyViewLutTexture);
        glBindTexture(GL_TEXTURE_2D, skyViewLutTexture);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R11F_G11F_B10F, 192, 108);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // -----------------------------------------------------------------
        // Render RayMarching.
        // -----------------------------------------------------------------
        GLuint renderRayMarchingPixelShader = sb7::shader::load("resources/shaders/render_raymarching.fs.glsl", GL_FRAGMENT_SHADER);

        renderRayMarchingProgram = glCreateProgram();

        glAttachShader(renderRayMarchingProgram, fullscreenQuadVertexShader);
        glAttachShader(renderRayMarchingProgram, renderRayMarchingPixelShader);

        glLinkProgram(renderRayMarchingProgram);

        uniforms.renderRayMarching.width = glGetUniformLocation(renderRayMarchingProgram, "width");
        uniforms.renderRayMarching.height = glGetUniformLocation(renderRayMarchingProgram, "height");
        uniforms.renderRayMarching.skyViewLutTexture = glGetUniformLocation(renderRayMarchingProgram, "skyViewLutTexture");

        glGenTextures(1, &renderRayMarchingTexture);
        glBindTexture(GL_TEXTURE_2D, renderRayMarchingTexture);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, info.windowWidth, info.windowHeight);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // -----------------------------------------------------------------
        // PostProcess
        // -----------------------------------------------------------------
        GLuint postProcessPixelShader = sb7::shader::load("resources/shaders/postprocess.fs.glsl", GL_FRAGMENT_SHADER);

        postProcessProgram = glCreateProgram();

        glAttachShader(postProcessProgram, fullscreenQuadVertexShader);
        glAttachShader(postProcessProgram, postProcessPixelShader);

        glLinkProgram(postProcessProgram);

        uniforms.postProcess.width = glGetUniformLocation(postProcessProgram, "width");
        uniforms.postProcess.height = glGetUniformLocation(postProcessProgram, "height");
        uniforms.postProcess.renderRayMarchingTexture = glGetUniformLocation(postProcessProgram, "renderRayMarchingTexture");

        // FBO.
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Setup atmospheric parameters.
        const float earthBottomRadius = 6360.0f;
        const float earthTopRadius = 6460.0f;
        const float earthRayleighScaleHeight = 8.0f;
        const float earthMieScaleHeight = 1.2f;
        const float PI = 3.14159265358979323846f;

        // Earth.
        atmosphericParameters.bottomRadius = earthBottomRadius;
        atmosphericParameters.topRadius = earthTopRadius;
        atmosphericParameters.groundAlbedo = { 0.0f, 0.0f, 0.0f };

        // Rayleigh scattering.
        atmosphericParameters.rayleighDensity.layers[0] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        atmosphericParameters.rayleighDensity.layers[1] = { 0.0f, 1.0f, -1.0f / earthRayleighScaleHeight , 0.0f, 0.0f };
        atmosphericParameters.rayleighScattering = { 0.005802f, 0.013558f, 0.033100f };

        // Mie scattering.
        atmosphericParameters.mieDensity.layers[0] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        atmosphericParameters.mieDensity.layers[1] = { 0.0f, 1.0f, -1.0f / earthMieScaleHeight , 0.0f, 0.0f };
        atmosphericParameters.mieScattering = { 0.003996f, 0.003996f, 0.003996f };
        atmosphericParameters.mieExtinction = { 0.004440f, 0.004440f, 0.004440f };
        atmosphericParameters.miePhaseFunctionG = 0.8f;

        // Ozon absorption.
        atmosphericParameters.absorptionDensity.layers[0] = { 25.0f, 0.0f, 0.0f, 1.0f / 15.0f, -2.0f / 3.0f };
        atmosphericParameters.absorptionDensity.layers[1] = { 0.0f, 0.0f, 0.0f, -1.0f / 15.0f, 8.0f / 3.0f };
        atmosphericParameters.absorptionExtinction = { 0.000650f, 0.001881f, 0.000085f };

        atmosphericParameters.multipleScatteringFactor = 1.0f;

        const double max_sun_zenith_angle = PI * 120.0 / 180.0; // (use_half_precision_ ? 102.0 : 120.0) / 180.0 * kPi;
    }

    virtual void render(double currentTime)
    {
        vmath::vec3 cameraPosition = vmath::vec3(0.0f, -1.0f, 0.5f);

        vmath::mat4 mv_matrix =
            vmath::lookat(cameraPosition, vmath::vec3(0.0f, 0.0f, 0.5f), vmath::vec3(0.0f, 0.0f, 1.0f));

        vmath::mat4 proj_matrix = vmath::perspective(60.0f,
            (float)info.windowWidth / (float)info.windowHeight,
            0.1f, 1000.0f);

        static const GLfloat green[] = { 0.0f, 0.25f, 0.0f, 1.0f };
        static const GLfloat black[] = { 0, 0, 0, 0 };
        static const GLfloat one = 1.0f;

        // -----------------------------------------------------------------
        // Update Atmospheric Parameters
        // -----------------------------------------------------------------
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, atmosphericParametersUniformBuffer);

        AtmosphericParametersUniformBuffer* atmosphericParametersBuffer =
            (AtmosphericParametersUniformBuffer*)glMapBufferRange(GL_UNIFORM_BUFFER, 0, sizeof(AtmosphericParametersUniformBuffer), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

        atmosphericParametersBuffer->absorptionExtinction = atmosphericParameters.absorptionExtinction;

        memcpy(atmosphericParametersBuffer->rayleighDensity, &atmosphericParameters.rayleighDensity, sizeof(atmosphericParameters.rayleighDensity));
        memcpy(atmosphericParametersBuffer->mieDensity, &atmosphericParameters.mieDensity, sizeof(atmosphericParameters.mieDensity));
        memcpy(atmosphericParametersBuffer->absorptionDensity, &atmosphericParameters.absorptionDensity, sizeof(atmosphericParameters.absorptionDensity));

        atmosphericParametersBuffer->miePhaseFunctionG = atmosphericParameters.miePhaseFunctionG;
        atmosphericParametersBuffer->rayleighScattering = atmosphericParameters.rayleighScattering;
        atmosphericParametersBuffer->mieScattering = atmosphericParameters.mieScattering;
        atmosphericParametersBuffer->mieAbsorption = vmath::max(atmosphericParameters.mieExtinction - atmosphericParameters.mieScattering, vmath::vec3(0.0f, 0.0f, 0.0f));
        atmosphericParametersBuffer->mieExtinction = atmosphericParameters.mieExtinction;
        atmosphericParametersBuffer->groundAlbedo = atmosphericParameters.groundAlbedo;
        atmosphericParametersBuffer->bottomRadius = atmosphericParameters.bottomRadius;
        atmosphericParametersBuffer->topRadius = atmosphericParameters.topRadius;

        atmosphericParametersBuffer->viewMatrix = mv_matrix;
        atmosphericParametersBuffer->projectionMatrix = proj_matrix;

        atmosphericParametersBuffer->cameraPos = cameraPosition;
        atmosphericParametersBuffer->multipleScatteringFactor = atmosphericParameters.multipleScatteringFactor;
        atmosphericParametersBuffer->sunDirection = vmath::vec3(0.0f, 0.900447130f, 0.434965521f); // z-up
        atmosphericParametersBuffer->globalLuminance = vmath::vec3(1.0f, 1.0f, 1.0f);

        glUnmapBuffer(GL_UNIFORM_BUFFER);

        static const GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0 };

        // -----------------------------------------------------------------
        // Render TransmittanceLUT.
        // -----------------------------------------------------------------
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, transmittanceLutTexture, 0);
        glViewport(0, 0, 256, 64);
        glDrawBuffers(1, drawBuffers);
        glClearBufferfv(GL_COLOR, 0, black);

        glUseProgram(transmittanceLutProgram);

        glUniform1i(uniforms.transmittanceLUT.width, 256);
        glUniform1i(uniforms.transmittanceLUT.height, 64);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glBindTexture(GL_TEXTURE_2D, 0);

        // -----------------------------------------------------------------
        // Render MultiscatteringLUT.
        // -----------------------------------------------------------------
        glUseProgram(multiScatteringLutProgram);

        glUniform1i(uniforms.multiScatteringLUT.transmittanceLutTexture, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, transmittanceLutTexture);

        glBindImageTexture(1, multiScatteringLutTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

        glDispatchCompute(32, 32, 1);

        glBindTexture(GL_TEXTURE_2D, 0);

        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        // -----------------------------------------------------------------
        // Render SkyViewLUT.
        // -----------------------------------------------------------------
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, skyViewLutTexture, 0);
        glViewport(0, 0, 192, 108);
        glDrawBuffers(1, drawBuffers);
        glClearBufferfv(GL_COLOR, 0, black);

        glUseProgram(skyViewLutProgram);

        glUniform1i(uniforms.skyviewLUT.transmittanceLutTexture, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, transmittanceLutTexture);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glBindTexture(GL_TEXTURE_2D, 0);

        // -----------------------------------------------------------------
        // Render RayMarching.
        // -----------------------------------------------------------------
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderRayMarchingTexture, 0);
        glViewport(0, 0, info.windowWidth, info.windowHeight);
        glDrawBuffers(1, drawBuffers);
        glClearBufferfv(GL_COLOR, 0, black);

        glUseProgram(renderRayMarchingProgram);

        glUniform1i(uniforms.renderRayMarching.width, info.windowWidth);
        glUniform1i(uniforms.renderRayMarching.height, info.windowHeight);

        glUniform1i(uniforms.renderRayMarching.skyViewLutTexture, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, skyViewLutTexture);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glBindTexture(GL_TEXTURE_2D, 0);

        // -----------------------------------------------------------------
        // Final PostProcess.
        // -----------------------------------------------------------------
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, info.windowWidth, info.windowHeight);
        glClearBufferfv(GL_COLOR, 0, black);

        glUseProgram(postProcessProgram);

        glUniform1i(uniforms.postProcess.width, info.windowWidth);
        glUniform1i(uniforms.postProcess.height, info.windowHeight);

        glUniform1i(uniforms.postProcess.renderRayMarchingTexture, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, renderRayMarchingTexture);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    virtual void shutdown()
    {
        glDeleteVertexArrays(1, &vao);

        glDeleteProgram(transmittanceLutProgram);
        glDeleteProgram(multiScatteringLutProgram);
        glDeleteProgram(skyViewLutProgram);
        glDeleteProgram(renderRayMarchingProgram);

        glDeleteFramebuffers(1, &fbo);

        glDeleteTextures(1, &transmittanceLutTexture);
        glDeleteTextures(1, &multiScatteringLutTexture);
        glDeleteTextures(1, &skyViewLutTexture);
        glDeleteTextures(1, &renderRayMarchingTexture);
    }

private:
    GLuint vao;

    // Shader Programs.
    GLuint transmittanceLutProgram;
    GLuint multiScatteringLutProgram;
    GLuint skyViewLutProgram;
    GLuint renderRayMarchingProgram;
    GLuint postProcessProgram;

    // Textures.
    GLuint transmittanceLutTexture;
    GLuint multiScatteringLutTexture;
    GLuint skyViewLutTexture;
    GLuint renderRayMarchingTexture;

    GLuint atmosphericParametersUniformBuffer;

    GLuint fbo;

    struct
    {
        struct
        {
            GLint width;
            GLint height;
        } transmittanceLUT;

        struct
        {
            GLint transmittanceLutTexture;
            GLint outputTexture;
        } multiScatteringLUT;

        struct
        {
            GLint transmittanceLutTexture;
        } skyviewLUT;

        struct
        {
            GLint width;
            GLint height;
            GLint skyViewLutTexture;
        } renderRayMarching;

        struct
        {
            GLint width;
            GLint height;
            GLint renderRayMarchingTexture;
        } postProcess;
    } uniforms;

    struct DensityProfileLayer
    {
        float width;
        float expTerm;
        float expScale;
        float linearTerm;
        float constantTerm;
    };

    struct DensityProfile
    {
        DensityProfileLayer layers[2];
    };

    struct AtmosphericParameters
    {
        float bottomRadius;
        float topRadius;
        vmath::vec3 groundAlbedo;

        DensityProfile rayleighDensity;
        vmath::vec3 rayleighScattering;

        DensityProfile mieDensity;
        vmath::vec3 mieScattering;
        vmath::vec3 mieExtinction;
        float miePhaseFunctionG;

        DensityProfile absorptionDensity;
        vmath::vec3 absorptionExtinction;

        float multipleScatteringFactor;
    } atmosphericParameters;

    struct AtmosphericParametersUniformBuffer
    {
        vmath::vec3 absorptionExtinction;
        float miePhaseFunctionG;

        vmath::vec3 rayleighScattering;
        float bottomRadius;

        vmath::vec3 mieScattering;
        float topRadius;

        vmath::vec3 mieExtinction;
        float pad0;

        vmath::vec3 mieAbsorption;
        float pad1;

        vmath::vec3 groundAlbedo;
        float pad2;

        float rayleighDensity[12];
        float mieDensity[12];
        float absorptionDensity[12];

        vmath::mat4 viewMatrix;
        vmath::mat4 projectionMatrix;

        vmath::vec3 cameraPos;
        float multipleScatteringFactor;
        vmath::vec3 sunDirection;
        float pad3;
        vmath::vec3 globalLuminance;
        float pad4;
    };
};

DECLARE_MAIN(SkyAtmosphere)
