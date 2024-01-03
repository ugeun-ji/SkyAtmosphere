#version 430 core

layout(location = 0) out vec4 color;

uniform int width;
uniform int height;

uniform sampler2D renderRayMarchingTexture;

void main(void)
{
    vec2 uv = gl_FragCoord.xy / vec2(width, height);
    
    vec4 rgba = texture(renderRayMarchingTexture, uv);
    rgba /= rgba.aaaa;

    vec3 whitePoint = vec3(1.08241, 0.96756, 0.95003);
    float exposure = 10.0f;
    color = vec4(pow(vec3(1.0f, 1.0f, 1.0f) - exp(-rgba.rgb  / whitePoint * exposure), vec3(1.0f / 2.2f)), 1.0f);
}
