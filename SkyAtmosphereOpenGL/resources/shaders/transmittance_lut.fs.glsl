#version 430 core

layout(location = 0) out vec4 color;

uniform int width;
uniform int height;

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

struct SingleScatteringResult
{
    vec3 luminance;
    vec3 opticalDepth;
    vec3 transmittance;
    vec3 multiScatAs1;
    vec3 newMultiScatStep0Out;
    vec3 newMultiScatStep1Out;
};

void UvToLutTransmittanceParams(in AtmosphericParameters atmosphericParameters, in vec2 uv, out float viewHeight, out float viewZenithCosAngle)
{
    float h = sqrt(atmosphericParameters.topRadius * atmosphericParameters.topRadius - atmosphericParameters.bottomRadius * atmosphericParameters.bottomRadius);
    float rho = h * uv.y;
    viewHeight = sqrt(rho * rho + atmosphericParameters.bottomRadius * atmosphericParameters.bottomRadius);

    float d_min = atmosphericParameters.topRadius - viewHeight;
    float d_max = rho + h;
    float d = d_min + uv.x * (d_max - d_min);
    viewZenithCosAngle = d == 0.0f ? 1.0f : (h * h - rho * rho - d * d) / (2.0f * viewHeight * d);
    viewZenithCosAngle = clamp(viewZenithCosAngle, -1.0f, 1.0f);
}

vec2 LutTransmittanceParamsToUv(in AtmosphericParameters atmosphericParameters, in float viewHeight, in float viewZenithCosAngle)
{
    float H = sqrt(max(0.0f, atmosphericParameters.topRadius * atmosphericParameters.topRadius - atmosphericParameters.bottomRadius * atmosphericParameters.bottomRadius));
    float rho = sqrt(max(0.0f, viewHeight * viewHeight - atmosphericParameters.bottomRadius * atmosphericParameters.bottomRadius));

    float discriminant = viewHeight * viewHeight * (viewZenithCosAngle * viewZenithCosAngle - 1.0f) + atmosphericParameters.topRadius * atmosphericParameters.topRadius;
    float d = max(0.0f, (-viewHeight * viewZenithCosAngle + sqrt(discriminant)));

    float d_min = atmosphericParameters.topRadius - viewHeight;
    float d_max = rho + H;
    float x_mu = (d - d_min) / (d_max - d_min);
    float x_r = rho / H;

    return vec2(x_mu, x_r);
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

float RayleighPhase(float cosTheta)
{
    float factor = 3.0f / (16.0f * PI);
    return factor * (1.0f + cosTheta * cosTheta);
}

float CornetteShanksMiePhaseFunction(float g, float cosTheta)
{
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cosTheta * cosTheta) / pow(1.0 + g * g - 2.0 * g * -cosTheta, 1.5);
}

float hgPhase(float g, float cosTheta)
{
//#ifdef USE_CornetteShanks
    return CornetteShanksMiePhaseFunction(g, cosTheta);
//#else
    // Reference implementation (i.e. not schlick approximation). 
    // See http://www.pbr-book.org/3ed-2018/Volume_Scattering/Phase_Functions.html
//    float numer = 1.0f - g * g;
//    float denom = 1.0f + g * g + 2.0f * g * cosTheta;
//    return numer / (4.0f * PI * denom * sqrt(denom));
//#endif
}

struct MediumSampleRGB
{
    vec3 scattering;
    vec3 absorption;
    vec3 extinction;

    vec3 scatteringMie;
    vec3 absorptionMie;
    vec3 extinctionMie;

    vec3 scatteringRay;
    vec3 absorptionRay;
    vec3 extinctionRay;

    vec3 scatteringOzo;
    vec3 absorptionOzo;
    vec3 extinctionOzo;

    vec3 albedo;
};

float GetAlbedo(float scattering, float extinction)
{
    return scattering / max(0.001, extinction);
}

vec3 GetAlbedo(vec3 scattering, vec3 extinction)
{
    return scattering / max(vec3(0.001, 0.001, 0.001), extinction);
}

MediumSampleRGB SampleMediumRGB(in vec3 worldPos, in AtmosphericParameters atmosphericParameters)
{
    const float viewHeight = length(worldPos) - atmosphericParameters.bottomRadius;

    const float densityMie = exp(atmosphericParameters.mieDensityExpScale * viewHeight);
    const float densityRay = exp(atmosphericParameters.rayleighDensityExpScale * viewHeight);
    const float densityOzo = clamp(viewHeight < atmosphericParameters.absorptionDensity0LayerWidth ? 
        atmosphericParameters.absorptionDensity0LinearTerm * viewHeight + atmosphericParameters.absorptionDensity0ConstantTerm : 
        atmosphericParameters.absorptionDensity1LinearTerm * viewHeight + atmosphericParameters.absorptionDensity1ConstantTerm, 0.0f, 1.0f);

    MediumSampleRGB s;

    s.scatteringMie = densityMie * atmosphericParameters.mieScattering;
    s.absorptionMie = densityMie * atmosphericParameters.mieAbsorption;
    s.extinctionMie = densityMie * atmosphericParameters.mieExtinction;

    s.scatteringRay = densityRay * atmosphericParameters.rayleighScattering;
    s.absorptionRay = vec3(0.0f, 0.0f, 0.0f);
    s.extinctionRay = s.scatteringRay + s.absorptionRay;

    s.scatteringOzo = vec3(0.0f, 0.0f, 0.0f);
    s.absorptionOzo = densityOzo * atmosphericParameters.absorptionExtinction;
    s.extinctionOzo = s.scatteringOzo + s.absorptionOzo;

    s.scattering = s.scatteringMie + s.scatteringRay + s.scatteringOzo;
    s.absorption = s.absorptionMie + s.absorptionRay + s.absorptionOzo;
    s.extinction = s.extinctionMie + s.extinctionRay + s.extinctionOzo;
    s.albedo = GetAlbedo(s.scattering, s.extinction);

    return s;
}

vec3 CalculateOpticalDepth(in vec2 pixPos, in vec3 worldPos, in vec3 worldDir, in AtmosphericParameters atmosphericParameters, in float sampleCount, in float tMaxMax = 9000000.0f)
{
    vec3 opticalDepth = vec3(0.0f, 0.0, 0.0);

    vec3 earth0 = vec3(0.0f, 0.0f, 0.0f);
    float tBottom = raySphereIntersectNearest(worldPos, worldDir, earth0, atmosphericParameters.bottomRadius);
    float tTop = raySphereIntersectNearest(worldPos, worldDir, earth0, atmosphericParameters.topRadius);
    float tMax = 0.0f;

    // bottom +, top - : 발생하지 않음.
    if (tBottom < 0.0f)
    {
        if (tTop < 0.0f) // bottom -, top - : 대기권 바깥.
        {
            tMax = 0.0f;
            return opticalDepth;
        }
        else // bottom -, top + : 대기권 내에 있고 하늘 방향.
        {
            tMax = tTop;
        }
    }
    else
    {
        if (tTop > 0.0f) // bottom +, top + : 대기권 내에 있고 지면 방향.
        {
            tMax = min(tTop, tBottom);
        }
    }

    tMax = min(tMax, tMaxMax);

    float dt = tMax / sampleCount;

    const float sampleSegmentT = 0.3f;

    float t = 0.0f;

    for (float s = 0.0f; s < sampleCount; s += 1.0f)
    {
        float newT = tMax * (s + sampleSegmentT) / sampleCount;

        dt = newT - t;

        t = newT;

        vec3 P = worldPos + t * worldDir;

        MediumSampleRGB medium = SampleMediumRGB(P, atmosphericParameters);

        opticalDepth += medium.extinction * dt;
    }

    return opticalDepth;
}

void main(void)
{
    AtmosphericParameters atmosphericParameters = GetAtmosphericParameters();

    vec2 uv = gl_FragCoord.xy / vec2(width, height);
    uv.y = 1.0f - uv.y;

    // uv.x : 대기 높이, uv.y : 천정 고도각
    // 위 계산값으로 월드 위치, 월드 방향을 구함.
    float viewHeight;
    float viewZenithCosAngle;
    UvToLutTransmittanceParams(atmosphericParameters, uv, viewHeight, viewZenithCosAngle);

    vec3 worldPos = vec3(0.0f, 0.0f, viewHeight);
    vec3 worldDir = vec3(0.0f, sqrt(1.0f - viewZenithCosAngle * viewZenithCosAngle), viewZenithCosAngle);

    // extinction 값을 구하여 남아있는 Luminance 를 구함(총 Luminance * Transmittance).
    // skyview_lut.fs.glsl 참조 요.
    const float sampleCount = 40.0f;
    vec3 opticalDepth = CalculateOpticalDepth(gl_FragCoord.xy, worldPos, worldDir, atmosphericParameters, sampleCount);
    vec3 transmittance = exp(-opticalDepth);

    color = vec4(transmittance, 1.0f);
}
