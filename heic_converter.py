#!/usr/bin/env python3
"""Convert images to HEIC format using Wand with ImageMagick subprocess fallback."""

import os
import sys
import time
import subprocess
import argparse
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Literal, Protocol, TypeAlias
from collections.abc import Iterator, Sequence

try:
    from wand.image import Image as WandImage
    WAND_AVAILABLE = True
except ImportError:
    WAND_AVAILABLE = False

# Type aliases
ConversionMethod: TypeAlias = Literal["wand", "subprocess"]
FileExtension: TypeAlias = Literal["jpg", "jpeg", "bmp", "png"]


@dataclass(frozen=True)
class ConversionResult:
    """Result of an image conversion operation."""
    success: bool
    src: Path
    dst: Path | None = None
    method: ConversionMethod | None = None
    error: str | None = None
    duration: float = 0.0
    original_size: int = 0
    converted_size: int = 0


@dataclass
class ConversionStats:
    """Statistics for the conversion process."""
    total: int = 0
    successful: int = 0
    failed: int = 0
    method_counts: dict[ConversionMethod, int] = None

    def __post_init__(self) -> None:
        if self.method_counts is None:
            object.__setattr__(self, 'method_counts', {})

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return (self.successful / self.total * 100) if self.total > 0 else 0.0


class ImageConverter(Protocol):
    """Protocol for image conversion functions."""
    def __call__(self, src: Path, dst: Path) -> None: ...


def convert_with_wand(src: Path, dst: Path) -> None:
    """Convert image using Wand (ImageMagick Python binding)."""
    with WandImage(filename=str(src)) as img:
        img.format = 'heic'
        img.save(filename=str(dst))


def convert_with_subprocess(src: Path, dst: Path) -> None:
    """Convert image using ImageMagick subprocess."""
    result = subprocess.run(
        ['magick', str(src), str(dst)],
        capture_output=True,
        text=True,
        check=True
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stderr)


def get_converter() -> tuple[ConversionMethod, ImageConverter]:
    """Get the appropriate converter based on availability."""
    if WAND_AVAILABLE:
        return "wand", convert_with_wand
    return "subprocess", convert_with_subprocess


def convert_image(args: tuple[Path, bool]) -> ConversionResult:
    """
    Convert a single image to HEIC format.
    
    Args:
        args: Tuple of (source_path, remove_original)
    
    Returns:
        ConversionResult with operation details
    """
    src_path, remove_original = args
    dst_path = src_path.with_suffix('.heic')
    start_time = time.perf_counter()
    original_size = src_path.stat().st_size
    converted_size = 0
    
    # Try Wand first, fallback to subprocess
    converters: list[tuple[ConversionMethod, ImageConverter]] = []
    if WAND_AVAILABLE:
        converters.append(("wand", convert_with_wand))
    converters.append(("subprocess", convert_with_subprocess))
    
    last_error: str | None = None
    
    for method, converter in converters:
        try:
            converter(src_path, dst_path)
            converted_size = dst_path.stat().st_size
            
            if remove_original:
                src_path.unlink()
            
            duration = time.perf_counter() - start_time
            return ConversionResult(
                success=True,
                src=src_path,
                dst=dst_path,
                method=method,
                duration=duration,
                original_size=original_size,
                converted_size=converted_size
            )
        except Exception as e:
            last_error = f"{method}: {str(e)}"
            if method == converters[-1][0]:  # Last method
                break
            continue
    
    duration = time.perf_counter() - start_time
    return ConversionResult(
        success=False,
        src=src_path,
        error=last_error,
        duration=duration,
        original_size=0,
        converted_size=0
    )


def get_image_files(
    directory: Path,
    extensions: Sequence[str],
    include_subfolders: bool
) -> Iterator[Path]:
    """
    Get all image files with specified extensions.
    
    Args:
        directory: Root directory to search
        extensions: List of file extensions to include
        include_subfolders: Whether to search recursively
    
    Yields:
        Path objects for matching image files
    """
    for ext in extensions:
        pattern = f"**/*.{ext}" if include_subfolders else f"*.{ext}"
        
        # Handle both lowercase and uppercase extensions
        yield from directory.glob(pattern)
        yield from directory.glob(pattern.upper())


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description='Convert images to HEIC format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('directory', type=Path, help='Directory containing images')
    parser.add_argument('-r', '--remove', action='store_true', 
                       help='Remove original files after conversion')
    parser.add_argument('-s', '--subfolders', action='store_true', 
                       help='Include subfolders')
    parser.add_argument('-w', '--workers', type=int, default=os.cpu_count(), 
                       help='Number of parallel workers')
    parser.add_argument('-e', '--extensions', nargs='+', 
                       default=['jpg', 'jpeg', 'bmp', 'png'],
                       help='File extensions to process')
    return parser


def format_size(size_bytes: float) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def print_summary(stats: ConversionStats, elapsed_time: float, total_size_saved: float) -> None:
    """Print conversion summary and statistics."""
    print("\n" + "=" * 50)
    print("CONVERSION SUMMARY")
    print("=" * 50)
    print(f"Total files: {stats.total}")
    print(f"Successful: {stats.successful} ({stats.success_rate:.1f}%)")
    print(f"Failed: {stats.failed}")
    print(f"Space saved: {format_size(total_size_saved)}")
    
    if stats.method_counts:
        print("\nMethods used:")
        for method, count in stats.method_counts.items():
            print(f"  {method}: {count} files")
    
    print(f"\nTime: {elapsed_time:.2f} seconds")
    if stats.total > 0:
        files_per_second = stats.total / elapsed_time
        print(f"Performance: {files_per_second:.2f} files/second")


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate directory
    if not args.directory.is_dir():
        print(f"Error: Directory '{args.directory}' does not exist", file=sys.stderr)
        return 1
    
    # Show available methods
    print("Available conversion methods:")
    print(f"  {'✓' if WAND_AVAILABLE else '✗'} Wand (ImageMagick binding)")
    print("  ✓ Subprocess (ImageMagick fallback)")
    print()
    
    # Get image files
    image_files = list(get_image_files(args.directory, args.extensions, args.subfolders))
    total_files = len(image_files)
    
    if total_files == 0:
        print("No image files found to convert")
        return 0
    
    print(f"Found {total_files} image files to convert")
    print(f"Extensions: {args.extensions}")
    if args.subfolders:
        print("Including subfolders")
    if args.remove:
        print("Original files will be removed")
    print(f"Using {args.workers} parallel workers\n")
    
    # Calculate total size of original files for space-saving summary
    original_total_size = sum(f.stat().st_size for f in image_files if f.exists())

    start_time = time.perf_counter()
    stats = ConversionStats()
    total_converted = 0
    total_size_saved: float = 0.0

    try:
        # Process files in parallel
        conversion_args = [(file_path, args.remove) for file_path in image_files]

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {
                executor.submit(convert_image, arg): arg[0]
                for arg in conversion_args
            }

            for i, future in enumerate(as_completed(future_to_file), 1):
                result = future.result()

                # Update statistics
                stats = ConversionStats(
                    total=stats.total + 1,
                    successful=stats.successful + (1 if result.success else 0),
                    failed=stats.failed + (0 if result.success else 1),
                    method_counts=stats.method_counts
                )

                if result.method:
                    stats.method_counts[result.method] = stats.method_counts.get(result.method, 0) + 1

                # Display progress
                status = "✓" if result.success else "✗"
                method_info = f" ({result.method})" if result.method else ""
                error_info = f" - {result.error}" if result.error else ""

                print(f"[{i}/{total_files}] {status} {result.src}{method_info}{error_info}")

                # Progress bar
                progress = (i / total_files) * 100
                print(f"Progress: {progress:.1f}% ({stats.successful} successful, "
                      f"{stats.failed} failed)", end='\r', flush=True)

                if result.success:
                    total_converted += 1
                    total_size_saved += (result.original_size - result.converted_size)

    except KeyboardInterrupt:
        print("\n\nConversion interrupted by user")
    finally:
        elapsed_time = time.perf_counter() - start_time
        print_summary(stats, elapsed_time, total_size_saved)

    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
