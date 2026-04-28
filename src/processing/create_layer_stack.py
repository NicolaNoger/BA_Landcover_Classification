"""
Erweitert vorhandene Buffered-Image-Stapel um zusätzliche Kanäle:

1. Normalisierte LAZ-Dateien -> nDSM, intensity, number_of_returns Raster
2. Resampling auf Ziel-Grid der vorhandenen TIFFs
3. NDVI aus Red + NIR berechnen
4. Ergebnis: bestehende Bänder + intensity + number_of_returns + NDVI
"""

import numpy as np
import laspy
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import maximum_filter
from pathlib import Path


def create_lidar_rasters_from_laz(
    laz_files: list[Path],
    output_dir: Path,
    resolution: float = 0.5,
    bounds: tuple = None,
    chunk_size: int = 5_000_000,
) -> dict[str, Path]:
    """
    Erstellt nDSM-, intensity- und number_of_returns-Raster aus
    normalisierten LAZ-Dateien.

    Args:
        laz_files: Liste der normalisierten LAZ-Dateien
        output_dir: Ausgabeordner für die Raster
        resolution: Rasterauflösung in Metern
        bounds: (xmin, ymin, xmax, ymax) oder None für automatisch
        chunk_size: Anzahl Punkte pro Chunk beim Einlesen

    Returns:
        Dict mit Ausgabepfaden der Lidar-Raster
    """
    print("Erstelle Lidar-Raster (nDSM, intensity, returns)...")

    if not laz_files:
        raise ValueError("Keine normalisierten LAZ-Dateien gefunden.")

    total_points = 0

    # Bounds aus Headern lesen (speicherschonend)
    if bounds is None:
        xmin, ymin = np.inf, np.inf
        xmax, ymax = -np.inf, -np.inf

        for laz_file in laz_files:
            with laspy.open(laz_file) as reader:
                mins = reader.header.mins
                maxs = reader.header.maxs
                xmin = min(xmin, mins[0])
                ymin = min(ymin, mins[1])
                xmax = max(xmax, maxs[0])
                ymax = max(ymax, maxs[1])
                total_points += reader.header.point_count
    else:
        xmin, ymin, xmax, ymax = bounds
        for laz_file in laz_files:
            with laspy.open(laz_file) as reader:
                total_points += reader.header.point_count

    print(f"  Gesamtpunkte: {total_points:,}")

    # Rastergrösse berechnen
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))

    print(f"  Rastergrösse: {width} x {height} Pixel ({resolution}m)")

    # nDSM: Maximum-Höhe pro Pixel (für Dächer/Baumkronen)
    ndsm_temp = np.full((height, width), -np.inf, dtype=np.float32)
    intensity_sum = np.zeros((height, width), dtype=np.float64)
    intensity_count = np.zeros((height, width), dtype=np.uint32)
    returns_temp = np.full((height, width), -np.inf, dtype=np.float32)

    # Punkte chunk-weise verarbeiten, um RAM-Spitzen zu vermeiden
    for laz_file in laz_files:
        print(f"  Lese (chunked): {laz_file.name}")
        with laspy.open(laz_file) as reader:
            for points in reader.chunk_iterator(chunk_size):
                x = np.asarray(points.x)
                y = np.asarray(points.y)
                z = np.asarray(points.z, dtype=np.float32)
                intensity = np.asarray(points.intensity, dtype=np.float32)

                if hasattr(points, "number_of_returns"):
                    num_returns = np.asarray(points.number_of_returns, dtype=np.float32)
                else:
                    num_returns = np.asarray(points.num_returns, dtype=np.float32)

                cols = ((x - xmin) / resolution).astype(np.int32)
                rows = ((ymax - y) / resolution).astype(np.int32)  # Y ist invertiert im Raster

                valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
                if not np.any(valid):
                    continue

                cols = cols[valid]
                rows = rows[valid]
                z = z[valid]
                intensity = intensity[valid]
                num_returns = num_returns[valid]

                np.maximum.at(ndsm_temp, (rows, cols), z)
                np.add.at(intensity_sum, (rows, cols), intensity)
                np.add.at(intensity_count, (rows, cols), 1)
                np.maximum.at(returns_temp, (rows, cols), num_returns)

    ndsm = np.full((height, width), np.nan, dtype=np.float32)
    intensity_raster = np.full((height, width), np.nan, dtype=np.float32)
    returns_raster = np.full((height, width), np.nan, dtype=np.float32)

    ndsm_valid = ndsm_temp > -np.inf
    ndsm[ndsm_valid] = ndsm_temp[ndsm_valid]

    intensity_valid = intensity_count > 0
    intensity_raster[intensity_valid] = (
        intensity_sum[intensity_valid] / intensity_count[intensity_valid]
    ).astype(np.float32)

    returns_valid = returns_temp > -np.inf
    returns_raster[returns_valid] = returns_temp[returns_valid]

    # Kleine Lücken füllen mit Maximum-Filter
    valid_mask = ~np.isnan(ndsm)
    if not valid_mask.all():
        # Fülle Lücken mit lokalem Maximum (3x3 Fenster)
        ndsm_filled = maximum_filter(np.nan_to_num(ndsm, nan=0), size=3)
        ndsm[~valid_mask] = ndsm_filled[~valid_mask]

    # Transform erstellen
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "ndsm": output_dir / "ndsm.tif",
        "intensity": output_dir / "intensity.tif",
        "returns": output_dir / "number_of_returns.tif",
    }

    raster_data = {
        "ndsm": ndsm,
        "intensity": intensity_raster,
        "returns": returns_raster,
    }

    for key, out_path in outputs.items():
        print(f"  Speichere: {out_path.name}")
        with rasterio.open(
            out_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs='EPSG:2056',
            transform=transform,
            nodata=np.nan,
            compress='LZW'
        ) as dst:
            dst.write(raster_data[key], 1)

    print(f"  nDSM Z-Bereich: {np.nanmin(ndsm):.2f} - {np.nanmax(ndsm):.2f} m")
    print(
        f"  Intensity Bereich: {np.nanmin(intensity_raster):.2f} - "
        f"{np.nanmax(intensity_raster):.2f}"
    )
    print(
        f"  Returns Bereich: {np.nanmin(returns_raster):.2f} - "
        f"{np.nanmax(returns_raster):.2f}"
    )

    return outputs


def enrich_existing_stack(
    stack_path: Path,
    intensity_path: Path,
    returns_path: Path,
    output_path: Path
) -> Path:
    """
    Erweitert vorhandenen Stack um intensity, number_of_returns und NDVI.

    Args:
        stack_path: Pfad zum vorhandenen Stack (mind. RGB + NIR)
        intensity_path: Pfad zum intensity-Raster
        returns_path: Pfad zum number_of_returns-Raster
        output_path: Ausgabepfad

    Returns:
        Pfad zum Layer-Stack
    """
    print(f"\nErweitere Stack für: {stack_path.name}")

    with rasterio.open(stack_path) as img:
        img_data = img.read()
        img_transform = img.transform
        img_width = img.width
        img_height = img.height
        descriptions = tuple(
            d if d is not None else f"Band_{i + 1}"
            for i, d in enumerate(img.descriptions)
        )

        print(f"  Input-Stack: {img_width}x{img_height}, {img.count} Bänder")

    # Intensity resampling
    with rasterio.open(intensity_path) as intensity_src:
        intensity_resampled = np.empty((img_height, img_width), dtype=np.float32)

        reproject(
            source=rasterio.band(intensity_src, 1),
            destination=intensity_resampled,
            src_transform=intensity_src.transform,
            src_crs=intensity_src.crs,
            dst_transform=img_transform,
            dst_crs='EPSG:2056',
            resampling=Resampling.bilinear
        )

        print(
            f"  Intensity Bereich: {np.nanmin(intensity_resampled):.2f} - "
            f"{np.nanmax(intensity_resampled):.2f}"
        )

    # Number-of-returns resampling
    with rasterio.open(returns_path) as returns_src:
        returns_resampled = np.empty((img_height, img_width), dtype=np.float32)

        reproject(
            source=rasterio.band(returns_src, 1),
            destination=returns_resampled,
            src_transform=returns_src.transform,
            src_crs=returns_src.crs,
            dst_transform=img_transform,
            dst_crs='EPSG:2056',
            resampling=Resampling.nearest
        )

        print(
            f"  Returns Bereich: {np.nanmin(returns_resampled):.2f} - "
            f"{np.nanmax(returns_resampled):.2f}"
        )

    desc_lower = [d.lower() for d in descriptions]

    def find_idx(candidates: tuple[str, ...], fallback: int) -> int:
        for candidate in candidates:
            if candidate in desc_lower:
                return desc_lower.index(candidate)
        return min(fallback, img_data.shape[0] - 1)

    nir_idx = find_idx(("nir",), 0)
    red_idx = find_idx(("red",), 1)
    green_idx = find_idx(("green",), 2)
    blue_idx = find_idx(("blue",), 3)
    ndsm_idx = find_idx(("ndsm_cm", "ndsm"), 4)

    red = img_data[red_idx].astype(np.float32)
    nir = img_data[nir_idx].astype(np.float32)
    denom = nir + red
    ndvi = np.divide(
        nir - red,
        denom,
        out=np.zeros_like(denom, dtype=np.float32),
        where=denom != 0,
    )

    # Skalierung für uint16-Stack
    intensity_scaled = np.clip(np.nan_to_num(intensity_resampled, nan=0), 0, 65535).astype(np.uint16)
    returns_scaled = np.clip(np.nan_to_num(returns_resampled, nan=0), 0, 65535).astype(np.uint16)
    ndvi_scaled = np.clip(((ndvi + 1.0) * 5000.0), 0, 10000).astype(np.uint16)

    base_stack = np.stack(
        [
            img_data[nir_idx],
            img_data[red_idx],
            img_data[green_idx],
            img_data[blue_idx],
            img_data[ndsm_idx],
        ],
        axis=0,
    ).astype(np.uint16)

    extra_channels = np.stack([intensity_scaled, returns_scaled, ndvi_scaled], axis=0)
    stack = np.concatenate([base_stack, extra_channels], axis=0)

    print(f"  Stack Shape neu: {stack.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=img_height,
        width=img_width,
        count=stack.shape[0],
        dtype='uint16',
        crs='EPSG:2056',
        transform=img_transform,
        compress='LZW'
    ) as dst:
        dst.write(stack)
        dst.descriptions = (
            'NIR',
            'Red',
            'Green',
            'Blue',
            'nDSM_cm',
            'intensity',
            'number_of_returns',
            'NDVI_x10000',
        )

    print(f"  Gespeichert: {output_path.name}")

    return output_path


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent

    # Pfade
    normalized_folder = project_root / "data" / "lidar" / "normalized"
    aerial_folder = project_root / "data" / "aerial" / "Buffered_images" / "images"
    output_folder = project_root / "data" / "processed" / "layer_stacks_enriched"
    lidar_raster_folder = project_root / "data" / "processed" / "lidar_rasters"

    # 1. Lidar-Raster erstellen (falls noch nicht vorhanden)
    lidar_paths = {
        "ndsm": lidar_raster_folder / "ndsm.tif",
        "intensity": lidar_raster_folder / "intensity.tif",
        "returns": lidar_raster_folder / "number_of_returns.tif",
    }

    missing_lidar = [p for p in lidar_paths.values() if not p.exists()]
    if missing_lidar:
        laz_files = [
            f for f in sorted(normalized_folder.glob("normalized_*.laz"))
            if not f.name.endswith('.copc.laz')
        ]
        print(f"Gefunden: {len(laz_files)} normalisierte LAZ-Dateien")
        lidar_paths = create_lidar_rasters_from_laz(
            laz_files,
            lidar_raster_folder,
            resolution=0.5,
        )
    else:
        print("Lidar-Raster existieren bereits.")

    # 2. Vorhandene Buffered-Stacks erweitern
    print("\n" + "=" * 50)
    print("Erweitere bestehende Stacks...")
    print("=" * 50)

    existing_stacks = sorted(aerial_folder.glob("*.tif"))
    print(f"Gefunden: {len(existing_stacks)} TIFFs")

    for img_path in existing_stacks:
        output_path = output_folder / f"{img_path.stem}_plus_lidar_ndvi.tif"
        enrich_existing_stack(
            img_path,
            lidar_paths["intensity"],
            lidar_paths["returns"],
            output_path,
        )

    print("\n" + "=" * 50)
    print("Fertig!")
