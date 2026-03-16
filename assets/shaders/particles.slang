import "shared.slang";

struct CircleMaterial {
    float3 center;
}

struct CameraData {
	column_major float4x4 view;
	column_major float4x4 proj;
}

struct InstanceData {
    CircleMaterial* instances;
	CameraData* camera;
}

struct VertexOutput {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
    float4 color : TEXCOORD1;
}

[shader("vertex")]
VertexOutput vertexMain(
    uint vert_id : SV_VertexID, 
    uint inst_id : SV_InstanceID,
    GraphicsPipelineData<InstanceData> pipeline)
{
    VertexOutput vert_out;
    float radius = 5.0;
    
    CircleMaterial mat = pipeline.vert->instances[inst_id];
    CameraData* cam = pipeline.vert->camera;

    float2 corners[4] = {
        float2(-1.0,  1.0), float2( 1.0,  1.0),
        float2(-1.0, -1.0), float2( 1.0, -1.0)
    };
    uint quad_indices[6] = { 0, 2, 1, 1, 2, 3 };
    float2 offset = corners[quad_indices[vert_id % 6]];

    float4 viewCenter = mul(cam->view, float4(mat.center, 1.0));
    
    float4 viewPos = viewCenter + float4(offset * radius, 0.0, 0.0);

    vert_out.pos = mul(cam->proj, viewPos);
    
    vert_out.uv = offset;
    vert_out.color = float4(1, 1, 1, 1);
    
    return vert_out;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input, GraphicsPipelineData<InstanceData> pipeline) : SV_Target
{
    float d = length(input.uv) - 1.0;
    float alpha = 1.0 - smoothstep(-0.01, 0.01, d);

    return input.color * alpha;
}
