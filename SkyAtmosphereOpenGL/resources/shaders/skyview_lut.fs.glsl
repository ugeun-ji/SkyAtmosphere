#version 430 core

layout(location = 0) out vec4 color;

uniform int width;
uniform int height;

layout(binding = 0) uniform sampler2D transmittanceLutTexture;

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

SingleScatteringResult IntegrateScatteredLuminance(in vec2 pixPos, in vec3 worldPos, in vec3 worldDir, in vec3 sunDir, in AtmosphericParameters atmosphericParameters,
    in bool ground, in float sampleCountOrig, in float depthBufferValue, in bool variableSampleCount, in bool mieRayPhase, in float tMaxMax = 9000000.0f)
{
    SingleScatteringResult result;

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
            return result;
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

    float sampleCount = sampleCountOrig;
    float sampleCountFloor = sampleCountOrig;
    float tMaxFloor = tMax;
    if (variableSampleCount)
    {
        //sampleCount = lerp(RayMarchMinMaxSPP.x, RayMarchMinMaxSPP.y, clamp(tMax * 0.01f, 0.0f, 1.0f));
        sampleCount = mix(4, 14, clamp(tMax * 0.01, 0.0f, 1.0f)); // lerp.
        sampleCountFloor = floor(sampleCount);
        tMaxFloor = tMax * sampleCountFloor / sampleCount;
    }

    float dt = tMax / sampleCount;

    // Phase Functions.
    const float uniformPhase = 1.0 / (4.0 * PI);
    float cosTheta = dot(sunDir, worldDir);
    float miePhaseValue = hgPhase(atmosphericParameters.miePhaseG, -cosTheta);
    float rayleighPhaseValue = RayleighPhase(cosTheta);

    vec3 globalL = globalLuninance;

    vec3 L = vec3(0.0f, 0.0f, 0.0f);
    vec3 throughput = vec3(1.0f, 1.0f, 1.0f);
    vec3 opticalDepth = vec3(0.0f, 0.0f, 0.0f);
    float t = 0.0f;
    float tPrev = 0.0f;
    const float sampleSegmentT = 0.3f;

    for (float s = 0.0f; s < sampleCount; s += 1.0f)
    {
        if (variableSampleCount)
        {
            float t0 = (s) / sampleCountFloor;
            float t1 = (s + 1.0f) / sampleCountFloor;

            t0 = t0 * t0;
            t1 = t1 * t1;

            t0 = tMaxFloor * t0;

            if (t1 > 1.0f)
            {
                t1 = tMax;
            }
            else
            {
                t1 = tMaxFloor * t1;
            }

            t = t0 + (t1 - t0) * sampleSegmentT;
            dt = t1 - t0;
        }
        else
        {
            float newT = tMax * (s + sampleSegmentT) / sampleCount;
            dt = newT - t;
            t = newT;
        }

        vec3 P = worldPos + t * worldDir;

        MediumSampleRGB medium = SampleMediumRGB(P, atmosphericParameters);
        const vec3 sampleOpticalDepth = medium.extinction * dt;
        const vec3 sampleTransmittance = exp(-sampleOpticalDepth);
        opticalDepth += sampleOpticalDepth;

        float height = length(P);
        const vec3 upVector = P / height;
        float sunZenithCosAngle = dot(sunDir, upVector);
        vec2 uv = LutTransmittanceParamsToUv(atmosphericParameters, height, sunZenithCosAngle);
        vec3 transmittanceToSun = texture(transmittanceLutTexture, uv).rgb;

        vec3 phaseTimesScattering;
        if (mieRayPhase)
        {
            phaseTimesScattering = medium.scatteringMie * miePhaseValue + medium.scatteringRay * rayleighPhaseValue;
        }
        else
        {
            phaseTimesScattering = medium.scattering * uniformPhase;
        }

        float tEarth = raySphereIntersectNearest(P, sunDir, earth0 + PLANET_RADIUS_OFFSET * upVector, atmosphericParameters.bottomRadius);

        float earthShadow = tEarth >= 0.0f ? 0.0f : 1.0f;

        float multiScatteredLuminance = 0.0f;

        float shadow = 1.0f;

        vec3 S = globalL * (earthShadow * shadow * transmittanceToSun * phaseTimesScattering + multiScatteredLuminance * medium.scattering);

        vec3 MS = medium.scattering * 1;
        vec3 MSint = (MS - MS * sampleTransmittance) / medium.extinction;

        result.multiScatAs1 += throughput * MSint;

        vec3 newMS;

        newMS = earthShadow * transmittanceToSun * medium.scattering * uniformPhase * 1;
        result.newMultiScatStep0Out += throughput * (newMS - newMS * sampleTransmittance) / medium.extinction;

        newMS = medium.scattering * uniformPhase * multiScatteredLuminance;
        result.newMultiScatStep1Out += throughput * (newMS - newMS * sampleTransmittance) / medium.extinction;

        vec3 Sint = (S - S * sampleTransmittance) / medium.extinction;
        L += throughput * Sint;
        throughput *= sampleTransmittance;

        tPrev = t;
    }

    if (ground && tMax == tBottom && tBottom > 0.0)
    {
        vec3 P = worldPos + tBottom * worldDir;
        float height = length(P);

        const vec3 upVector = P / height;
        float sunZenithCosAngle = dot(sunDir, upVector);

        vec2 uv = LutTransmittanceParamsToUv(atmosphericParameters, height, sunZenithCosAngle);
        vec3 transmittanceToSun = texture(transmittanceLutTexture, uv).rgb;

        const float NdotL = clamp(dot(normalize(upVector), normalize(sunDir)), 0.0f, 1.0f);
        L += globalL * transmittanceToSun * throughput * NdotL * atmosphericParameters.groundAlbedo / PI;
    }

    result.luminance = L;
    result.opticalDepth = opticalDepth;
    result.transmittance = throughput;

    return result;
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
}

void UvToSkyViewLutParams(in AtmosphericParameters atmosphericParameters, in float viewHeight, in vec2 uv, out float viewZenithCosAngle, out float lightViewCosAngle)
{
    uv = vec2(FromSubUvsToUnit(uv.x, 192.0f), FromSubUvsToUnit(uv.y, 108.0f));

    // 
    float vhorizon = sqrt(viewHeight * viewHeight - atmosphericParameters.bottomRadius * atmosphericParameters.bottomRadius);
    float cosBeta = vhorizon / viewHeight;
    float beta = acos(cosBeta);
    float zenithHorizonAngle = PI - beta;

    if (uv.y < 0.5f) // 지평선 아래
    {
        float coord = 2.0f * uv.y;
        coord = 1.0f - coord;
        coord *= coord; // Non Linear.
        coord = 1.0f - coord;
        viewZenithCosAngle = cos(zenithHorizonAngle * coord);
    }
    else // 지평선 위
    {
        float coord = uv.y * 2.0f - 1.0f;
        coord *= coord; // Non Linear.
        viewZenithCosAngle = cos(zenithHorizonAngle + beta * coord);
    }

    float coord = uv.x;
    coord *= coord;
    lightViewCosAngle = -(coord * 2.0f - 1.0f);
}

bool MoveToTopAtmosphere(inout vec3 worldPos, in vec3 worldDir, in float atmosphereTopRadius)
{
    float viewHeight = length(worldPos);

    if (viewHeight > atmosphereTopRadius)
    {
        float tTop = raySphereIntersectNearest(worldPos, worldDir, vec3(0.0f, 0.0f, 0.0f), atmosphereTopRadius);
        if (tTop >= 0.0f)
        {
            vec3 upVector = worldPos / viewHeight;
            vec3 upOffset = upVector * -PLANET_RADIUS_OFFSET;
            worldPos = worldPos + worldDir * tTop + upOffset;
        }
        else
        {
            return false;
        }
    }

    return true;
}

void main(void)
{
    AtmosphericParameters atmosphericParameters = GetAtmosphericParameters();

    vec2 uv = gl_FragCoord.xy / vec2(192.0f, 108.0f);
    uv.y = 1.0f - uv.y;

    vec3 worldPos = cameraPos + vec3(0.0f, 0.0f, atmosphericParameters.bottomRadius);
    float viewHeight = length(worldPos);

    float viewZenithCosAngle;
    float lightViewCosAngle;
    UvToSkyViewLutParams(atmosphericParameters, viewHeight, uv, viewZenithCosAngle, lightViewCosAngle);

    vec3 upVector = worldPos / viewHeight;
    float sunZenithCosAngle = dot(upVector, sunDirection);
    vec3 sunDir = normalize(vec3(sqrt(1.0f - sunZenithCosAngle * sunZenithCosAngle), 0.0f, sunZenithCosAngle));

    worldPos = vec3(0.0f, 0.0f, viewHeight);

    float viewZenithSinAngle = sqrt(1 - viewZenithCosAngle * viewZenithCosAngle);

    vec3 worldDir = vec3(viewZenithSinAngle * lightViewCosAngle, viewZenithSinAngle * sqrt(1.0f - lightViewCosAngle * lightViewCosAngle), viewZenithCosAngle);

    if (!MoveToTopAtmosphere(worldPos, worldDir, atmosphericParameters.topRadius))
    {
        color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    const bool ground = false;
    const float sampleCount = 30.0f;
    const float depthBufferValue = -1.0f;
    const bool variableSampleCount = true;
    const bool mieRayPhase = true;

    SingleScatteringResult result = IntegrateScatteredLuminance(gl_FragCoord.xy, worldPos, worldDir, sunDir, atmosphericParameters, ground, sampleCount, depthBufferValue, variableSampleCount, mieRayPhase);

    vec3 luminance = result.luminance;

    color = vec4(luminance, 1.0f);
}
