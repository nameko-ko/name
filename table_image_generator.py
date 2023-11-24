from PIL import Image, ImageDraw, ImageFont

def create_table_image(table_data, table_name):
    num_columns = len(table_data[0])
    num_rows = len(table_data)
    cell_width = 150
    cell_height = 30
    padding = 10
    font = ImageFont.load_default()

    image_width = num_columns * cell_width
    image_height = (num_rows + 1) * cell_height  # +1 for header

    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)

    # テーブル名を描画
    draw.text(
        (padding, padding),
        table_name,
        font=font,
        fill="black",
    )
    for i, row in enumerate(table_data):
        for j, cell in enumerate(row):
            x = j * cell_width
            y = (i + 1) * cell_height  # +1 for header
            draw.rectangle(
                [(x, y), (x + cell_width, y + cell_height)], outline="black", fill="white"
            )
            draw.text(
                (x + padding, y + padding),
                str(cell),
                font=font,
                fill="black",
            )

    return image


def find_min_max_avg_table(data_list):
    if not data_list:
        return None

    min_value = min(data_list)
    max_value = max(data_list)
    avg_value = sum(data_list) / len(data_list)

    min_index = data_list.index(min_value)
    max_index = data_list.index(max_value)

    table = [
        ["Property", "Value", "Index"],
        ["Minimum", min_value, min_index],
        ["Maximum", max_value, max_index],
        ["Average", avg_value, "N/A"]
    ]

    return table
