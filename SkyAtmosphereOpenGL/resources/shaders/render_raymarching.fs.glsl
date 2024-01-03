#version 430 core

layout(location = 0) out vec4 color;

uniform int width;
uniform int height;

uniform sampler2D skyViewLutTexture;

#define PI 3.1415926535897932384626433832795f

#define PLANET_RADIUS_OFFSET 0.01f

layout(std140) uniform AtmosphericParametersBuffer
{
    vec3 absorptionExtinction;
    float miePhaseFunctionG;

    vec3 rayleighScattering;
    float bottomRadius;

    vec3 mieScattering;
    float topRadius;

    vec3 mieExtinction;
    float pad0;

    vec3 mieAbsorption;
    float pad1;

    vec3 groundAlbedo;
    float pad2;

    vec4 rayleighDensity[3];
    vec4 mieDensity[3];
    vec4 absorptionDensity[3];

    mat4 viewMatrix;
    mat4 projectionMatrix;

    vec3 cameraPos;
    float multipleScatteringFactor;
    vec3 sunDirection;
    float pad3;
    vec3 globalLuninance;
    float pad4;
};

struct AtmosphericParameters
{
    float bottomRadius;
    float topRadius;

    float rayleighDensityExpScale;
    vec3 rayleighScattering;

    float mieDensityExpScale;
    vec3 mieScattering;
    vec3 mieExtinction;
    vec3 mieAbsorption;
    float miePhaseG;

    float absorptionDensity0LayerWidth;
    float absorptionDensity0ConstantTerm;
    float absorptionDensity0LinearTerm;
    float absorptionDensity1ConstantTerm;
    float absorptionDensity1LinearTerm;
    vec3 absorptionExtinction;

    vec3 groundAlbedo;
};

AtmosphericParameters GetAtmosphericParameters()
{
    AtmosphericParameters parameters;

    parameters.absorptionExtinction = absorptionExtinction;

    // Bruneton2017
    parameters.rayleighDensityExpScale = rayleighDensity[1].w;
    parameters.mieDensityExpScale = mieDensity[1].w;
    parameters.absorptionDensity0LayerWidth = absorptionDensity[0].x;
    parameters.absorptionDensity0ConstantTerm = absorptionDensity[1].x;
    parameters.absorptionDensity0LinearTerm = absorptionDensity[0].w;
    parameters.absorptionDensity1ConstantTerm = absorptionDensity[2].y;
    parameters.absorptionDensity1LinearTerm = absorptionDensity[2].x;

    parameters.miePhaseG = miePhaseFunctionG;
    parameters.rayleighScattering = rayleighScattering;
    parameters.mieScattering = mieScattering;
    parameters.mieAbsorption = mieAbsorption;
    parameters.mieExtinction = mieExtinction;
    parameters.groundAlbedo = groundAlbedo;
    parameters.bottomRadius = bottomRadius;
    parameters.topRadius = topRadius;

    return parameters;
}

// - r0: ray origin
// - rd: normalized ray direction
// - s0: sphere center
// - sR: sphere radius
// - Returns distance from r0 to first intersecion with sphere,
//   or -1.0 if no intersection.
float raySphereIntersectNearest(vec3 r0, vec3 rd, vec3 s0, float sR)
{
    float a = dot(rd, rd);
    vec3 s0_r0 = r0 - s0;
    float b = 2.0 * dot(rd, s0_r0);
    float c = dot(s0_r0, s0_r0) - (sR * sR);
    float delta = b * b - 4.0 * a * c;
    if (delta < 0.0 || a == 0.0)
    {
        return -1.0;
    }
    float sol0 = (-b - sqrt(delta)) / (2.0 * a);
    float sol1 = (-b + sqrt(delta)) / (2.0 * a);
    if (sol0 < 0.0 && sol1 < 0.0)
    {
        return -1.0;
    }
    if (sol0 < 0.0)
    {
        return max(0.0, sol1);
    }
    else if (sol1 < 0.0)
    {
        return max(0.0, sol0);
    }
    return max(0.0, min(sol0, sol1));
}

float FromUnitToSubUvs(float u, float resolution) { return (u + 0.5f / resolution) * (resolution / (resolution + 1.0f)); }
float FromSubUvsToUnit(float u, float resolution) { return (u - 0.5f / resolution) * (resolution / (resolution - 1.0f)); }

void SkyViewLutParamsToUv(in AtmosphericParameters atmosphericParameters, in bool intersectGround, in float viewZenithCosAngle, in float lightViewCosAngle, in float viewHeight, out vec2 uv)
{
    float vhorizon = sqrt(viewHeight * viewHeight - atmosphericParameters.bottomRadius * atmosphericParameters.bottomRadius);
    float cosBeta = vhorizon / viewHeight;
    float beta = acos(cosBeta);
    float zenithHorizonAngle = PI - beta;

    if (!intersectGround) // 지평선 위
    {
        float coord = acos(viewZenithCosAngle) / zenithHorizonAngle; // 0 ~ 1.
        coord = 1.0f - coord; // 0 ~ 1(지평선 ~ Up Vector).
        coord = sqrt(coord);  // Non Linear.
        coord = 1.0f - coord; // 원복. 0 ~ 1(Up Vector ~ 지평선).
        uv.y = coord * 0.5f; // 지평선 위 : 0.5 ~ 1.0. OpenGL Tex 기준. https://www.puredevsoftware.com/blog/2018/03/17/texture-coordinates-d3d-vs-opengl/
    }
    else // 지평선 아래
    {
        float coord = (acos(viewZenithCosAngle) - zenithHorizonAngle) / beta;
        coord = sqrt(coord);
        uv.y = coord * 0.5f + 0.5f;
    }

    // lightViewCosAngle : -1(180도) ~ 1(0도).
    float coord = -lightViewCosAngle * 0.5f + 0.5f; // 정규화
    coord = sqrt(coord);
    uv.x = coord;

    uv = vec2(FromUnitToSubUvs(uv.x, 192.0f), FromUnitToSubUvs(uv.y, 108.0f));
    uv.y = 1.0f - uv.y;
}

vec3 GetSunLuminance(vec3 worldPos, vec3 worldDir, float planetRadius)
{
    if (dot(worldDir, sunDirection) > cos(0.5 * 0.505 * 3.14159 / 180.0))
    {
        float t = raySphereIntersectNearest(worldPos, worldDir, vec3(0.0f, 0.0f, 0.0f), planetRadius);
        if (t < 0.0f)
        {
            const vec3 sunLuminance = vec3(1000000.0, 1000000.0, 1000000.0);

            return sunLuminance;
        }
    }

    return vec3(0.0f, 0.0f, 0.0f);
}

void main(void)
{
    AtmosphericParameters atmosphericParameters = GetAtmosphericParameters();

    vec2 pixPos = gl_FragCoord.xy;

    vec3 clipSpace = vec3((pixPos / vec2(width, height)) * vec2(2.0, 2.0) - vec2(1.0, 1.0), 1.0);

    vec4 viewPos = inverse(projectionMatrix) * vec4(clipSpace, 1.0);

    vec3 worldDir = normalize(inverse(mat3(viewMatrix)) * (viewPos.xyz / viewPos.w));

    vec3 worldPos = cameraPos + vec3(0.0f, 0.0f, atmosphericParameters.bottomRadius);

    float viewHeight = length(worldPos);

    vec3 luminance = vec3(0.0f, 0.0f, 0.0f);

    if (viewHeight < atmosphericParameters.topRadius)
    {
        vec3 upVector = normalize(worldPos);

        float viewZenithCosAngle = dot(worldDir, upVector);

        vec3 sideVector = normalize(cross(upVector, worldDir));

        vec3 forwardVector = normalize(cross(sideVector, upVector));
        
        vec2 lightOnPlane = vec2(dot(sunDirection, forwardVector), dot(sunDirection, sideVector));
        lightOnPlane = normalize(lightOnPlane);

        float lightViewCosAngle = lightOnPlane.x;

        bool bGroundIntersected = raySphereIntersectNearest(worldPos, worldDir, vec3(0.0f, 0.0f, 0.0f), atmosphericParameters.bottomRadius) >= 0.0f;

        vec2 uv; 
        SkyViewLutParamsToUv(atmosphericParameters, bGroundIntersected, viewZenithCosAngle, lightViewCosAngle, viewHeight, uv);

        vec3 sunLuminance = GetSunLuminance(worldPos, worldDir, atmosphericParameters.bottomRadius);

        luminance = vec4(texture(skyViewLutTexture, uv).rgb + sunLuminance, 1.0f).rgb;
    }

    color = vec4(luminance, 1.0f);
}
