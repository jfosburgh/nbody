#!/bin/bash

# Configuration
SLANG_FILE="assets/shaders/particles.slang"
OUTPUT_DIR="assets/shaders"
COMPILER="slangc"

# Check if slangc is in the PATH
if ! command -v $COMPILER &> /dev/null
then
    echo "Error: $COMPILER could not be found. Please ensure Slang is installed and in your PATH."
    exit 1
fi

echo "Compiling $SLANG_FILE..."

# 1. Compile Vertex Shader
# -target spirv: Output Vulkan-compatible SPIR-V
# -entry vertexMain: The name of the function in your slang file
# -stage vertex: Explicitly tell Slang this is the vertex stage
$COMPILER $SLANG_FILE \
    -target spirv \
    -entry vertexMain \
    -stage vertex \
    -o "$OUTPUT_DIR/particle.vert.spv"

if [ $? -eq 0 ]; then
    echo "Successfully generated particle.vert.spv"
else
    echo "Failed to compile vertex shader."
    exit 1
fi

# 2. Compile Fragment Shader
$COMPILER $SLANG_FILE \
    -target spirv \
    -entry fragmentMain \
    -stage fragment \
    -o "$OUTPUT_DIR/particle.frag.spv"

if [ $? -eq 0 ]; then
    echo "Successfully generated particle.frag.spv"
else
    echo "Failed to compile fragment shader."
    exit 1
fi

echo "All shaders compiled successfully."
