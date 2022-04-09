#version 330 core
out vec4 FragColor;

in VS_OUT
{
    vec2 texCoords;
}
fsIn;

uniform sampler2D colorBuffer;

void main()
{
    FragColor = texture(colorBuffer, fsIn.texCoords).rgba;
}