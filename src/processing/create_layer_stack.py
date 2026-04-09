"""
Erstellt den 5-Band Layer-Stack: RGB + NIR + nDSM

1. Normalisierte LAZ-Dateien → nDSM-Raster
2. Swissimage (4 Bänder) + nDSM = 5-Band-Stack
"""

import numpy as np
import laspy
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import maximum_filter
from pathlib import Path


def create_ndsm_from_laz(
    laz_files: list[Path],
    output_path: Path,
    resolution: float = 0.5,
    bounds: tuple = None
) -> Path:
    """
    Erstellt ein nDSM-Raster aus normalisierten LAZ-Dateien.

    Args:
        laz_files: Liste der normalisierten LAZ-Dateien
        output_path: Ausgabepfad für das nDSM
        resolution: Rasterauflösung in Metern
        bounds: (xmin, ymin, xmax, ymax) oder None für automatisch

    Returns:
        Pfad zum nDSM-Raster
    """
    print("Erstelle nDSM aus normalisierten LAZ-Dateien...")

    # Alle Punkte laden und Bounds bestimmen
    all_x, all_y, all_z = [], [], []

    for laz_file in laz_files:
        print(f"  Lese: {laz_file.name}")
        las = laspy.read(laz_file)
        all_x.append(las.x)
        all_y.append(las.y)
        all_z.append(las.z)

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    z = np.concatenate(all_z)

    print(f"  Gesamtpunkte: {len(x):,}")

    # Bounds bestimmen
    if bounds is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, ymin, xmax, ymax = bounds

    # Rastergrösse berechnen
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))

    print(f"  Rastergrösse: {width} x {height} Pixel ({resolution}m)")

    # Punkte zu Pixel-Indizes
    cols = ((x - xmin) / resolution).astype(int)
    rows = ((ymax - y) / resolution).astype(int)  # Y ist invertiert im Raster

    # Nur gültige Indizes
    valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    cols = cols[valid]
    rows = rows[valid]
    z = z[valid]

    # nDSM: Maximum-Höhe pro Pixel (für Dächer/Baumkronen)
    ndsm = np.full((height, width), np.nan, dtype=np.float32)

    # Verwende np.maximum.at für effizientes Maximum
    # Erst mit -inf initialisieren für Maximum-Berechnung
    ndsm_temp = np.full((height, width), -np.inf, dtype=np.float32)
    np.maximum.at(ndsm_temp, (rows, cols), z)

    # Zurück zu NaN wo keine Daten (blieb -inf)
    ndsm[ndsm_temp > -np.inf] = ndsm_temp[ndsm_temp > -np.inf]

    # Kleine Lücken füllen mit Maximum-Filter
    valid_mask = ~np.isnan(ndsm)
    if not valid_mask.all():
        # Fülle Lücken mit lokalem Maximum (3x3 Fenster)
        ndsm_filled = maximum_filter(np.nan_to_num(ndsm, nan=0), size=3)
        ndsm[~valid_mask] = ndsm_filled[~valid_mask]

    # Transform erstellen
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    # Speichern
    print(f"  Speichere: {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
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
        dst.write(ndsm, 1)

    print(f"  nDSM Z-Bereich: {np.nanmin(ndsm):.2f} - {np.nanmax(ndsm):.2f} m")

    return output_path


def create_layer_stack(
    swissimage_path: Path,
    ndsm_path: Path,
    output_path: Path
) -> Path:
    """
    Kombiniert Swissimage (4 Bänder) mit nDSM zu einem 5-Band-Stack.

    Args:
        swissimage_path: Pfad zum Swissimage (RGB+NIR)
        ndsm_path: Pfad zum nDSM-Raster
        output_path: Ausgabepfad

    Returns:
        Pfad zum Layer-Stack
    """
    print(f"\nErstelle Layer-Stack für: {swissimage_path.name}")

    # Swissimage öffnen
    with rasterio.open(swissimage_path) as img:
        img_data = img.read()  # Shape: (4, H, W)
        img_transform = img.transform
        img_bounds = img.bounds
        img_width = img.width
        img_height = img.height
        img_dtype = img.dtypes[0]

        print(f"  Swissimage: {img_width}x{img_height}, {img.count} Bänder")

    # nDSM öffnen und auf Swissimage-Grid resampling
    with rasterio.open(ndsm_path) as ndsm:
        # Resample nDSM auf Swissimage-Auflösung und Bounds
        ndsm_resampled = np.empty((1, img_height, img_width), dtype=np.float32)

        reproject(
            source=rasterio.band(ndsm, 1),
            destination=ndsm_resampled[0],
            src_transform=ndsm.transform,
            src_crs=ndsm.crs,
            dst_transform=img_transform,
            dst_crs='EPSG:2056',
            resampling=Resampling.bilinear
        )

        print(f"  nDSM resampled: {ndsm_resampled.shape}")
        print(f"  nDSM Bereich: {np.nanmin(ndsm_resampled):.2f} - {np.nanmax(ndsm_resampled):.2f} m")

    # nDSM auf uint16 skalieren (wie Swissimage)
    # Annahme: Höhen 0-100m -> 0-10000 (Faktor 100, 1cm Auflösung)
    ndsm_scaled = np.clip(ndsm_resampled * 100, 0, 65535).astype(np.uint16)
    ndsm_scaled = np.nan_to_num(ndsm_scaled, nan=0)

    # Stack erstellen: RGB + NIR + nDSM
    stack = np.concatenate([img_data, ndsm_scaled], axis=0)

    print(f"  Stack Shape: {stack.shape} (R, G, B, NIR, nDSM)")

    # Speichern
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=img_height,
        width=img_width,
        count=5,
        dtype='uint16',
        crs='EPSG:2056',
        transform=img_transform,
        compress='LZW'
    ) as dst:
        dst.write(stack)
        dst.descriptions = ('Red', 'Green', 'Blue', 'NIR', 'nDSM_cm')

    print(f"  Gespeichert: {output_path.name}")

    return output_path


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent

    # Pfade
    normalized_folder = project_root / "data" / "lidar" / "normalized"
    aerial_folder = project_root / "data" / "aerial"
    output_folder = project_root / "data" / "processed" / "layer_stacks"
    ndsm_path = project_root / "data" / "processed" / "ndsm.tif"

    # 1. nDSM erstellen (falls noch nicht vorhanden)
    if not ndsm_path.exists():
        # Nur normalized_X.laz, nicht .copc.laz
        laz_files = [f for f in sorted(normalized_folder.glob("normalized_*.laz"))
                     if not f.name.endswith('.copc.laz')]
        print(f"Gefunden: {len(laz_files)} normalisierte LAZ-Dateien")
        create_ndsm_from_laz(laz_files, ndsm_path, resolution=0.5)
    else:
        print(f"nDSM existiert bereits: {ndsm_path}")

    # 2. Layer-Stacks für jedes Swissimage erstellen
    print("\n" + "=" * 50)
    print("Erstelle Layer-Stacks...")
    print("=" * 50)

    swissimages = sorted(aerial_folder.glob("cliped_*_image.tif"))
    print(f"Gefunden: {len(swissimages)} Swissimages")

    for img_path in swissimages:
        output_path = output_folder / f"{img_path.stem}_stack.tif"
        create_layer_stack(img_path, ndsm_path, output_path)

    print("\n" + "=" * 50)
    print("Fertig!")
