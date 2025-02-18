def get_width_and_height_number_of_tiles(image, tile_size):
    height = image.shape[0]
    width = image.shape[1]
    height_num_tiles = height // tile_size
    if height % tile_size > 0:
        height_num_tiles += 1
    width_num_tiles = width // tile_size
    if width % tile_size > 0:
        width_num_tiles += 1
    return width_num_tiles, height_num_tiles


def get_tile_height_and_width_indexes(image, tile_index, tile_size):
    width_num_tiles, height_num_tiles = get_width_and_height_number_of_tiles(
        image, tile_size
    )
    tile_position_width = tile_index % width_num_tiles
    tile_position_height = tile_index // width_num_tiles
    tile_width_index = tile_position_width * tile_size
    tile_height_index = tile_position_height * tile_size
    if tile_position_width == width_num_tiles - 1:
        tile_width_index = image.shape[1] - tile_size
    if tile_position_height == height_num_tiles - 1:
        tile_height_index = image.shape[0] - tile_size
    return tile_height_index, tile_width_index


def get_tile(image, tile_height_index, tile_width_index, tile_size, mask=False):
    if mask:
        return image[
            tile_height_index : tile_height_index + tile_size,
            tile_width_index : tile_width_index + tile_size,
        ]
    return image[
        tile_height_index : tile_height_index + tile_size,
        tile_width_index : tile_width_index + tile_size,
        :,
    ]
