"""
LiDAR Normalisierung - Subtrahiert DTM-Höhen von Punktwolken-Z-Werten.

Dieses Skript normalisiert LAZ-Punktwolken, indem es die Geländehöhe
(aus einem DTM-Raster) von den Z-Koordinaten abzieht.
Ergebnis: Höhe über Grund für jeden Punkt.
"""

import numpy as np
import laspy
import rasterio
from pathlib import Path
from typing import Union


def normalize_laz_with_dtm(
    laz_path: Union[str, Path],
    dtm_path: Union[str, Path],
    output_path: Union[str, Path] = None,
) -> Path:
    """
    Normalisiert eine LAZ-Datei mit einem DTM-Raster.

    Args:
        laz_path: Pfad zur LAZ-Eingabedatei
        dtm_path: Pfad zum DTM-Raster (TIF)
        output_path: Pfad für die Ausgabedatei (optional, Standard: *_normalized.laz)

    Returns:
        Pfad zur normalisierten LAZ-Datei
    """
    laz_path = Path(laz_path)
    dtm_path = Path(dtm_path)

    if output_path is None:
        # .copc entfernen falls vorhanden
        stem = laz_path.stem
        if stem.endswith(".copc"):
            stem = stem[:-5]
        output_path = laz_path.parent / f"{stem}_normalized.laz"
    else:
        output_path = Path(output_path)

    # LAZ-Datei lesen
    print(f"Lese LAZ: {laz_path.name}")
    las = laspy.read(laz_path)

    # Koordinaten extrahieren
    x = las.x
    y = las.y
    z = las.z.copy()

    print(f"  Punkte: {len(x):,}")
    print(f"  Z-Bereich original: {z.min():.2f} - {z.max():.2f} m")

    # DTM öffnen und Höhen samplen
    print(f"Lese DTM: {dtm_path.name}")
    with rasterio.open(dtm_path) as dtm:
        # Prüfen ob CRS übereinstimmen (Warnung wenn nicht)
        dtm_crs = dtm.crs
        print(f"  DTM CRS: {dtm_crs}")
        print(f"  DTM Bounds: {dtm.bounds}")

        # Koordinaten zu Pixel-Indizes transformieren
        rows, cols = rasterio.transform.rowcol(dtm.transform, x, y)
        rows = np.array(rows)
        cols = np.array(cols)

        # DTM-Band lesen
        dtm_data = dtm.read(1)
        nodata = dtm.nodata

        # Punkte die ausserhalb des DTM liegen maskieren
        valid_mask = (
            (rows >= 0) & (rows < dtm.height) &
            (cols >= 0) & (cols < dtm.width)
        )

        # DTM-Höhen für gültige Punkte samplen
        terrain_heights = np.full(len(x), np.nan)
        terrain_heights[valid_mask] = dtm_data[rows[valid_mask], cols[valid_mask]]

        # NoData-Werte behandeln
        if nodata is not None:
            terrain_heights[terrain_heights == nodata] = np.nan

    # Statistiken über Sampling
    valid_samples = ~np.isnan(terrain_heights)
    print(f"  Gültige DTM-Samples: {valid_samples.sum():,} / {len(x):,} ({100*valid_samples.mean():.1f}%)")

    if not valid_samples.any():
        raise ValueError("Keine gültigen DTM-Werte gefunden! Prüfe CRS und räumliche Überlappung.")

    # Normalisierung: Z - DTM_Höhe
    z_normalized = z - terrain_heights

    # Punkte ohne gültige DTM-Höhe: Original-Z beibehalten
    z_normalized[~valid_samples] = z[~valid_samples]

    print(f"  Z-Bereich normalisiert: {np.nanmin(z_normalized[valid_samples]):.2f} - {np.nanmax(z_normalized[valid_samples]):.2f} m")

    # Neue LAZ-Datei erstellen (nicht COPC, sondern normales LAZ)
    # Wir müssen ein neues LasData-Objekt erstellen, da COPC nicht geschrieben werden kann
    header = laspy.LasHeader(point_format=las.header.point_format, version="1.4")
    header.offsets = las.header.offsets
    header.scales = las.header.scales

    # Neues LAS-Objekt mit den gleichen Punkten
    new_las = laspy.LasData(header)
    new_las.x = las.x
    new_las.y = las.y
    new_las.z = z_normalized

    # Alle anderen Attribute kopieren
    for dim in las.point_format.dimension_names:
        if dim not in ['X', 'Y', 'Z']:
            try:
                setattr(new_las, dim, getattr(las, dim))
            except Exception:
                pass  # Einige Dimensionen könnten nicht kopierbar sein

    # Speichern
    print(f"Speichere: {output_path.name}")
    new_las.write(output_path)

    return output_path


def normalize_folder(
    laz_folder: Union[str, Path],
    dtm_path: Union[str, Path],
    output_folder: Union[str, Path] = None,
    pattern: str = "*.laz"
) -> list[Path]:
    """
    Normalisiert alle LAZ-Dateien in einem Ordner.

    Args:
        laz_folder: Ordner mit LAZ-Dateien
        dtm_path: Pfad zum DTM-Raster
        output_folder: Ausgabeordner (optional, Standard: laz_folder/normalized/)
        pattern: Glob-Pattern für LAZ-Dateien

    Returns:
        Liste der Pfade zu normalisierten Dateien
    """
    laz_folder = Path(laz_folder)
    dtm_path = Path(dtm_path)

    if output_folder is None:
        output_folder = laz_folder / "normalized"
    else:
        output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    laz_files = sorted(laz_folder.glob(pattern))
    print(f"Gefunden: {len(laz_files)} LAZ-Dateien")
    print("-" * 50)

    results = []
    for i, laz_file in enumerate(laz_files, 1):
        print(f"\n[{i}/{len(laz_files)}] Verarbeite: {laz_file.name}")
        output_path = output_folder / f"normalized_{i}.laz"

        try:
            result = normalize_laz_with_dtm(laz_file, dtm_path, output_path)
            results.append(result)
        except Exception as e:
            print(f"  FEHLER: {e}")

    print("\n" + "=" * 50)
    print(f"Fertig! {len(results)}/{len(laz_files)} Dateien normalisiert.")

    return results


if __name__ == "__main__":
    # Pfade für dieses Projekt
    project_root = Path(__file__).parent.parent.parent

    laz_folder = project_root / "data" / "lidar"
    dtm_path = project_root / "data" / "Swissalti" / "swissalti_merged.tif"
    output_folder = project_root / "data" / "lidar" / "normalized"

    # Alle LAZ-Dateien normalisieren
    normalize_folder(
        laz_folder=laz_folder,
        dtm_path=dtm_path,
        output_folder=output_folder,
        pattern="*.copc.laz"
    )
