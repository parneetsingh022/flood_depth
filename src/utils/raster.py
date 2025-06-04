import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

def read_raster(file_path):
    """
    Reads a .tif raster file and returns the image data as a NumPy array
    along with the metadata (profile).
    
    Args:
        file_path (str): Path to the .tif file.
    
    Returns:
        tuple: (image_array, metadata)
    """
    with rasterio.open(file_path) as src:
        image_array = src.read(1)  # Read the first band
        metadata = src.profile
    return image_array, metadata


def plot_raster(raster, title="Raster Image", cmap='viridis', nodata=None, from_file=False, vmin=None, vmax=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import rasterio

    if from_file:
        with rasterio.open(raster) as src:
            image = src.read(1)
            if nodata is None:
                nodata = src.nodata
    else:
        image = raster

    # Always use masked array
    if nodata is not None:
        image = np.ma.masked_equal(image, nodata)
    else:
        image = np.ma.masked_invalid(image)

    # Auto-scale color limits
    if vmin is None:
        vmin = np.percentile(image.compressed(), 2)
    if vmax is None:
        vmax = np.percentile(image.compressed(), 98)

    plt.figure(figsize=(10, 6))
    img_plot = plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(img_plot, label='Pixel Values')
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.grid(False)
    plt.show()

def downscale_raster(input_path, output_path, target_width, target_height, resampling=Resampling.average):
    """
    Downscale a raster file to a target width and height using average resampling.

    Args:
        input_path (str): Path to the input .tif file.
        output_path (str): Path to save the downscaled output .tif.
        target_width (int): Desired width in pixels.
        target_height (int): Desired height in pixels.
        resampling: rasterio.enums.Resampling method (default: average).
    """
    with rasterio.open(input_path) as src:
        # Compute scale factors
        scale_x = src.width / target_width
        scale_y = src.height / target_height

        # Update transform and metadata
        transform = src.transform * src.transform.scale(scale_x, scale_y)
        kwargs = src.meta.copy()
        kwargs.update({
            'height': target_height,
            'width': target_width,
            'transform': transform,
            'compress': 'lzw'
        })

        # Read and resample
        data = src.read(
            out_shape=(src.count, target_height, target_width),
            resampling=resampling
        )

        # Write downscaled raster
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            dst.write(data)

