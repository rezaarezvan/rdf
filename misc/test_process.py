def customize_svg(input_file, output_file):
    with open(input_file, 'r') as file:
        svg_content = file.read()

    # Replace static colors with dynamic ones
    svg_content = svg_content.replace('fill="black"', 'fill="currentColor"')
    svg_content = svg_content.replace(
        'stroke="black"', 'stroke="currentColor"')

    # Add a style block
    style_block = '''
    <style>
        svg {
            color: currentColor;
            display: block;
            margin: auto;
            width: 85%;
            height: auto;
        }
        text {
            fill: currentColor;
        }
    </style>
    '''
    svg_content = svg_content.replace('<svg', f'<svg {style_block}')

    with open(output_file, 'w') as file:
        file.write(svg_content)


# Use the function
customize_svg("output.svg", "dynamic_output.svg")
