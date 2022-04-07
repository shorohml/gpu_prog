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
    vec3 color = texture(colorBuffer, fsIn.texCoords).rgb;
    FragColor = vec4(color, 1.0);
}