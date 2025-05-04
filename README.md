# HEIC Converter

Convert images to HEIC format using Wand (ImageMagick binding) or ImageMagick subprocess fallback.

## Why HEIC?

HEIC (High Efficiency Image Container) is a modern image format that offers significant advantages over traditional formats like JPEG:

- **50% smaller file sizes** compared to JPEG at the same quality level
- Better compression while maintaining high image quality
- Support for transparency and multiple images in a single file
- Modern format designed for today's high-resolution displays

## Usage

```bash
uv run --with wand heic_converter.py <directory> [options]
```

## Arguments

- `directory`: Directory containing images to convert (required)
- `-r, --remove`: Remove original files after successful conversion
- `-s, --subfolders`: Include subfolders in the search
- `-w, --workers`: Number of parallel workers (default: number of CPU cores)
- `-e, --extensions`: File extensions to process (default: jpg jpeg bmp png)

## Example

```bash
# Convert all images in current directory
uv run --with wand heic_converter.py .

# Convert images in a specific directory, including subfolders
uv run --with wand heic_converter.py /path/to/images --subfolders

# Convert only jpg files and remove originals
uv run --with wand heic_converter.py . --extensions jpg --remove
``` 