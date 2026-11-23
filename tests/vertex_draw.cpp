#include "desktop_gpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>
using Microsoft::WRL::ComPtr;

namespace {
std::vector<char> read_file(const char* path, const char* failure)
{
    std::ifstream stream(path, std::ios::binary);
    std::vector<char> bytes{std::istreambuf_iterator<char>(stream), {}};
    if (!stream.eof() || bytes.empty()) throw std::runtime_error(failure);
    return bytes;
}
}

// Draw three vertices without an input layout or vertex buffer, so the vertex shader
// receives only SV_VertexID, and compare the float target with a CPU reference.
void check_vertex_draw(ID3D12Device* device, const char* vertex, const char* pixel, const char* reference_path)
{
    constexpr UINT side = 32;
    constexpr float tolerance = 1.0f / 4096;
    const auto vs = read_file(vertex, "Cannot read translated vertex shader");
    const auto ps = read_file(pixel, "Cannot read pixel shader");
    const auto reference_bytes = read_file(reference_path, "Cannot read vertex draw reference");
    if (reference_bytes.size() != side * side * 4 * sizeof(float))
        throw std::runtime_error("Expected a 32x32 RGBA32F vertex draw reference");
    std::vector<float> reference(side * side * 4);
    std::memcpy(reference.data(), reference_bytes.data(), reference_bytes.size());

    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    ComPtr<ID3DBlob> blob;
    check(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr),
          "Serialize vertex draw root signature");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)),
          "Create vertex draw root signature");
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root.Get();
    desc.VS = {vs.data(), vs.size()};
    desc.PS = {ps.data(), ps.size()};
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    auto& blend = desc.BlendState.RenderTarget[0];
    blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
    blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
    blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    blend.LogicOp = D3D12_LOGIC_OP_NOOP;
    blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask = UINT_MAX;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = desc.SampleDesc.Count = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pipeline)), "Create translated vertex pipeline");

    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = texture.Height = side;
    texture.DepthOrArraySize = texture.MipLevels = texture.SampleDesc.Count = 1;
    texture.Format = desc.RTVFormats[0];
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    // A pixel the triangle misses keeps a value that no reference texel holds.
    D3D12_CLEAR_VALUE clear{texture.Format, {-1.0f, -1.0f, -1.0f, -1.0f}};
    auto target = resource(device, texture, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear);
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "Create vertex draw target heap");
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, handle);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 size = 0;
    device->GetCopyableFootprints(&texture, 0, 1, 0, &footprint, nullptr, nullptr, &size);
    auto readback = make_buffer(device, size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    list->SetGraphicsRootSignature(root.Get());
    const D3D12_VIEWPORT viewport{0, 0, side, side, 0, 1};
    const D3D12_RECT scissor{0, 0, side, side};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &handle, FALSE, nullptr);
    list->ClearRenderTargetView(handle, clear.Color, 0, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
    transition(list, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = target.Get();
    dst.pResource = readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = footprint;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    commands.submit_and_wait(device);

    void* mapped = nullptr;
    check(readback->Map(0, nullptr, &mapped), "Map vertex draw result");
    const char* rows = static_cast<const char*>(mapped) + footprint.Offset;
    unsigned mismatches = 0;
    float max_error = 0;
    for (UINT y = 0; y < side; ++y) {
        for (UINT x = 0; x < side; ++x) {
            float texel[4];
            std::memcpy(texel, rows + y * footprint.Footprint.RowPitch + x * sizeof(texel), sizeof(texel));
            const float* expected = reference.data() + (y * side + x) * 4;
            bool equal = true;
            for (int c = 0; c < 4; ++c) {
                const float error = std::fabs(texel[c] - expected[c]);
                equal = equal && error <= tolerance;
                if (error > max_error || error != error) max_error = error;
            }
            if (!equal && !mismatches++)
                std::printf("first_vertex_mismatch x=%u y=%u got=(%.9g, %.9g, %.9g, %.9g) "
                            "expected=(%.9g, %.9g, %.9g, %.9g)\n", x, y, texel[0], texel[1], texel[2],
                            texel[3], expected[0], expected[1], expected[2], expected[3]);
        }
    }
    const D3D12_RANGE empty{};
    readback->Unmap(0, &empty);
    std::printf("vertex_pixels=%u mismatches=%u max_error=%.9g\n", side * side, mismatches, max_error);
    if (mismatches) throw std::runtime_error("Vertex draw reference mismatch");
}
