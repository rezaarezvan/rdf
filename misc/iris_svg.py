import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import io
import re


def create_dynamic_plot():
    print("\n=== Starting Dynamic Plot Generation ===")

    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)
    print("✓ Loaded Iris dataset")

    # Use unique, recognizable colors as placeholders
    SETOSA_COLOR = '#FF0001'      # Unique red
    VERSICOLOR_COLOR = '#00FF02'  # Unique green
    VIRGINICA_COLOR = '#0000FF'   # Unique blue
    TEXT_COLOR = '#000000'        # Black for initial text/lines

    print("\nPlaceholder Colors:")
    print(f"Setosa:     {SETOSA_COLOR}")
    print(f"Versicolor: {VERSICOLOR_COLOR}")
    print(f"Virginica:  {VIRGINICA_COLOR}")
    print(f"Text:       {TEXT_COLOR}")

    # Create figure with transparent background
    fig = plt.figure(facecolor='none')
    ax = plt.gca()
    ax.set_facecolor('none')

    # Create histograms with placeholder colors
    plt.hist(X[y == 0, 2], color=SETOSA_COLOR, alpha=0.5, label='setosa')
    plt.hist(X[y == 1, 2], color=VERSICOLOR_COLOR,
             alpha=0.5, label='versicolor')
    plt.hist(X[y == 2, 2], color=VIRGINICA_COLOR, alpha=0.5, label='virginica')

    plt.title('Histogram of Petal Length', color=TEXT_COLOR)
    plt.xlabel('Petal Length (cm)', color=TEXT_COLOR)
    plt.ylabel('Frequency', color=TEXT_COLOR)

    # Style the legend and other elements
    leg = plt.legend()
    for text in leg.get_texts():
        text.set_color(TEXT_COLOR)

    # Style the spines and ticks
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

    print("\n✓ Created plot with placeholder colors")

    # Save to SVG string
    output = io.StringIO()
    plt.savefig(output, format='svg', bbox_inches='tight', transparent=True)
    plt.close()

    svg_content = output.getvalue()
    original_length = len(svg_content)
    print(f"\nOriginal SVG length: {original_length} characters")

    # Add CSS variables and styling
    style_defs = """
        <defs>
            <style>
                /* Light mode colors */
                :root {
                    --setosa-color: #d62728;
                    --versicolor-color: #2ca02c;
                    --virginica-color: #1f77b4;
                }

                /* Dark mode colors */
                .dark {
                    --setosa-color: #FF4A98;
                    --versicolor-color: #0AFAFA;
                    --virginica-color: #7F83FF;
                }

                .setosa {
                    fill: var(--setosa-color) !important;
                    opacity: 0.5;
                }

                .versicolor {
                    fill: var(--versicolor-color) !important;
                    opacity: 0.5;
                }

                .virginica {
                    fill: var(--virginica-color) !important;
                    opacity: 0.5;
                }

                /* Make all text inherit color */
                text, .text {
                    fill: currentColor !important;
                    stroke: none !important;
                }

                /* Make all lines inherit color */
                .axis line, .axis path {
                    stroke: currentColor !important;
                }

                /* Special handling for legend */
                .legend text {
                    fill: currentColor !important;
                }

                /* Make all default black strokes use currentColor */
                [stroke="#000000"] {
                    stroke: currentColor !important;
                }
            </style>
        </defs>
    """

    print("\nStarting SVG modifications:")

    # Insert style definitions
    svg_content = re.sub(r'(<svg[^>]*>)', rf'\1{style_defs}', svg_content)
    print("✓ Inserted style definitions")

    # Replace histogram colors with classes
    def replace_color_with_class(content, color, class_name):
        # Handle both style="fill: color" and fill="color" formats
        patterns = [
            rf'style="fill: {color}[^"]*"',
            rf'fill="{color}"'
        ]
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, content))
            content = re.sub(pattern, f'class="{class_name}"', content)
        return content, count

    # Process colors
    svg_content, setosa_count = replace_color_with_class(
        svg_content, SETOSA_COLOR.lower(), "setosa")
    print(f"✓ Replaced {setosa_count} setosa color instances")

    svg_content, versi_count = replace_color_with_class(
        svg_content, VERSICOLOR_COLOR.lower(), "versicolor")
    print(f"✓ Replaced {versi_count} versicolor color instances")

    svg_content, virg_count = replace_color_with_class(
        svg_content, VIRGINICA_COLOR.lower(), "virginica")
    print(f"✓ Replaced {virg_count} virginica color instances")

    print("\nProcessing text and line elements:")

    # Add classes to text elements
    text_patterns = [
        (r'<g id="([^"]*text[^"]*)"', r'<g id="\1" class="text"'),
        (r'<g id="matplotlib\.axis[^"]*"', r'<g class="axis"'),
        (r'<g id="legend[^"]*"', r'<g class="legend"')
    ]

    for pattern, replacement in text_patterns:
        count = len(re.findall(pattern, svg_content))
        if count > 0:
            svg_content = re.sub(pattern, replacement, svg_content)
            print(f"✓ Added class to {count} {
                  replacement.split('class=')[1]} elements")

    # Replace stroke colors - handle both style attribute and direct stroke attribute
    stroke_patterns = [
        r'stroke="#000000"',                           # Direct stroke attribute
        # Style attribute with stroke
        r'style="[^"]*stroke: ?#000000[^"]*"',
        # Style attribute without space
        r'style="[^"]*stroke:#000000[^"]*"'
    ]

    total_stroke_count = 0
    for pattern in stroke_patterns:
        count = len(re.findall(pattern, svg_content))
        total_stroke_count += count
        if 'style=' in pattern:
            # For style attributes, preserve other styles
            svg_content = re.sub(
                r'(style="[^"]*?)stroke: ?#000000([^"]*")',
                r'\1stroke: currentColor\2',
                svg_content
            )
        else:
            # For direct stroke attributes
            svg_content = re.sub(pattern, 'stroke="currentColor"', svg_content)

    print(f"✓ Replaced {total_stroke_count} stroke colors with currentColor")

    final_length = len(svg_content)
    print(f"\nFinal SVG length: {final_length} characters")
    print(f"Size difference: {final_length - original_length} characters")

    print("\n=== Dynamic Plot Generation Complete ===")

    return svg_content


if __name__ == "__main__":
    svg_content = create_dynamic_plot()
    with open('dynamic_iris_plot.svg', 'w') as f:
        f.write(svg_content)
