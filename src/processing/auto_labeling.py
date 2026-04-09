"""
Automatic labeling using SLIC superpixels + K-Means clustering.

1. Create superpixels from the 5-band layer stack (RGB + NIR + nDSM)
2. Extract features per superpixel (mean values)
3. Cluster superpixels using K-Means (test multiple k values)
4. Analyze which AV classes fall into which clusters
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from skimage.segmentation import slic
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


def create_superpixels(
    stack_path: Path,
    n_segments: int = 5000,
    compactness: float = 1.0  # Lower = more edge-following, higher = more square
) -> tuple[np.ndarray, dict]:
    """
    Create SLIC superpixels from the layer stack.

    Args:
        stack_path: Path to 5-band layer stack (R, G, B, NIR, nDSM)
        n_segments: Approximate number of superpixels
        compactness: Balance between color and spatial proximity
                     - Low (0.1-1): Irregular, edge-following superpixels
                     - High (10-100): Compact, square-ish superpixels

    Returns:
        segments: Label image with superpixel IDs
        metadata: Dict with transform, crs, shape info
    """
    print(f"Creating superpixels for: {stack_path.name}")

    with rasterio.open(stack_path) as src:
        # Read all bands
        data = src.read()  # Shape: (5, H, W)
        transform = src.transform
        crs = src.crs

        # Transpose to (H, W, C) for skimage
        img = np.transpose(data, (1, 2, 0)).astype(np.float32)

        # Normalize each channel to 0-1 for SLIC
        for i in range(img.shape[2]):
            channel = img[:, :, i]
            vmin, vmax = np.nanpercentile(channel, [2, 98])
            if vmax > vmin:
                img[:, :, i] = np.clip((channel - vmin) / (vmax - vmin), 0, 1)
            else:
                img[:, :, i] = 0

        # Replace NaN with 0
        img = np.nan_to_num(img, nan=0)

    print(f"  Image shape: {img.shape}")
    print(f"  Running SLIC with n_segments={n_segments}, compactness={compactness}")

    # Run SLIC with sigma for slight smoothing (helps with noisy data)
    segments = slic(
        img,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1.0,  # Slight Gaussian smoothing
        channel_axis=2,
        start_label=0,
        enforce_connectivity=True  # Ensure connected superpixels
    )

    n_actual = len(np.unique(segments))
    print(f"  Created {n_actual} superpixels")

    metadata = {
        'transform': transform,
        'crs': crs,
        'shape': segments.shape,
        'stack_path': stack_path
    }

    return segments, metadata


def extract_superpixel_features(
    stack_path: Path,
    segments: np.ndarray
) -> np.ndarray:
    """
    Extract mean feature values for each superpixel.
    Uses scipy.ndimage for fast vectorized computation.

    Args:
        stack_path: Path to layer stack
        segments: Superpixel label image

    Returns:
        features: Array of shape (n_superpixels, n_features)
    """
    from scipy import ndimage

    print("Extracting superpixel features...")

    with rasterio.open(stack_path) as src:
        data = src.read().astype(np.float32)  # (5, H, W)

    n_superpixels = segments.max() + 1
    n_bands = data.shape[0]

    # Use scipy.ndimage.mean for fast computation
    labels = np.arange(n_superpixels)
    features = np.zeros((n_superpixels, n_bands), dtype=np.float32)

    for band in range(n_bands):
        features[:, band] = ndimage.mean(data[band], labels=segments, index=labels)

    # Handle any remaining NaN
    features = np.nan_to_num(features, nan=0)

    print(f"  Features shape: {features.shape}")
    print(f"  Feature names: R, G, B, NIR, nDSM")

    return features


def cluster_superpixels(
    features: np.ndarray,
    k_values: list[int] = [10, 20, 30, 40, 50]
) -> dict[int, np.ndarray]:
    """
    Cluster superpixels using K-Means with multiple k values.

    Args:
        features: Superpixel features
        k_values: List of k values to test

    Returns:
        Dict mapping k -> cluster labels
    """
    print("\nClustering superpixels with K-Means...")

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    results = {}

    for k in k_values:
        print(f"  K={k}...", end=" ")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        inertia = kmeans.inertia_
        print(f"inertia={inertia:.0f}")
        results[k] = labels

    return results


def rasterize_av_classes(
    gpkg_path: Path,
    layer_name: str,
    metadata: dict,
    class_column: str = "Art"
) -> np.ndarray:
    """
    Rasterize AV vector data to match the layer stack grid.

    Args:
        gpkg_path: Path to GeoPackage
        layer_name: Layer name in GeoPackage
        metadata: Metadata from superpixel creation
        class_column: Column containing class IDs

    Returns:
        Rasterized class labels
    """
    print(f"\nRasterizing AV classes from: {layer_name}")

    # Load vector data
    gdf = gpd.read_file(gpkg_path, layer=layer_name)

    # Ensure same CRS
    if gdf.crs != metadata['crs']:
        gdf = gdf.to_crs(metadata['crs'])

    # Create shapes for rasterization
    shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[class_column])]

    # Rasterize
    av_raster = rasterize(
        shapes=shapes,
        out_shape=metadata['shape'],
        transform=metadata['transform'],
        fill=255,  # NoData value
        dtype=np.uint8
    )

    unique_classes = np.unique(av_raster[av_raster != 255])
    print(f"  Rasterized {len(unique_classes)} unique AV classes")

    return av_raster


def analyze_cluster_composition(
    segments: np.ndarray,
    cluster_labels: np.ndarray,
    av_raster: np.ndarray,
    av_class_names: dict[int, str],
    k: int
) -> dict:
    """
    Analyze which AV classes are dominant in each cluster.

    Args:
        segments: Superpixel label image
        cluster_labels: Cluster assignment per superpixel
        av_raster: Rasterized AV classes
        av_class_names: Mapping from class ID to name
        k: Number of clusters

    Returns:
        Dict with cluster composition stats
    """
    print(f"\n{'='*60}")
    print(f"Cluster composition analysis (K={k})")
    print('='*60)

    results = {}

    for cluster_id in range(k):
        # Get all superpixels in this cluster
        sp_in_cluster = np.where(cluster_labels == cluster_id)[0]

        # Create mask for all pixels in this cluster
        cluster_mask = np.isin(segments, sp_in_cluster)

        # Get AV classes in this cluster
        av_values = av_raster[cluster_mask]
        av_values = av_values[av_values != 255]  # Remove NoData

        if len(av_values) == 0:
            continue

        # Count class occurrences
        class_counts = Counter(av_values)
        total = sum(class_counts.values())

        # Get top classes
        top_classes = class_counts.most_common(5)

        results[cluster_id] = {
            'n_superpixels': len(sp_in_cluster),
            'n_pixels': total,
            'top_classes': [(av_class_names.get(c, f"Unknown_{c}"), count, count/total*100)
                           for c, count in top_classes]
        }

        # Print summary
        print(f"\nCluster {cluster_id}: {len(sp_in_cluster)} superpixels, {total:,} pixels")
        for name, count, pct in results[cluster_id]['top_classes']:
            # Shorten name for display
            short_name = name.split('.')[-1] if '.' in name else name
            print(f"  {pct:5.1f}% - {short_name}")

    return results


def save_cluster_raster(
    segments: np.ndarray,
    cluster_labels: np.ndarray,
    metadata: dict,
    output_path: Path,
    k: int
):
    """Save cluster labels as a raster."""
    # Map superpixel IDs to cluster IDs
    cluster_raster = cluster_labels[segments]

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=metadata['shape'][0],
        width=metadata['shape'][1],
        count=1,
        dtype='uint8',
        crs=metadata['crs'],
        transform=metadata['transform'],
        compress='LZW'
    ) as dst:
        dst.write(cluster_raster.astype(np.uint8), 1)

    print(f"Saved cluster raster: {output_path.name}")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent

    # Paths
    stack_folder = project_root / "data" / "processed" / "layer_stacks"
    gpkg_path = project_root / "data" / "geodata" / "DM01AVZH24LV95.gpkg"
    output_folder = project_root / "data" / "processed" / "clustering"
    output_folder.mkdir(parents=True, exist_ok=True)

    # AV class name mapping
    av_class_names = {
        0: "Gebaeude.Verwaltung",
        1: "Gebaeude.Wohngebaeude",
        2: "Gebaeude.Land_Forstwirtschaft",
        3: "Gebaeude.Verkehr",
        4: "Gebaeude.Handel",
        5: "Gebaeude.Industrie_Gewerbe",
        6: "Gebaeude.Gastgewerbe",
        7: "Gebaeude.Nebengebaeude",
        8: "befestigt.Strasse",
        9: "befestigt.Velo_Fussweg",
        10: "befestigt.Landwirtschaftsstrasse",
        11: "befestigt.Waldstrasse",
        12: "befestigt.Trottoir",
        13: "befestigt.Verkehrsinsel",
        14: "befestigt.Bahn",
        15: "befestigt.Flugplatz",
        16: "befestigt.Wasserbecken",
        17: "befestigt.Parkplatz",
        18: "befestigt.Hausumschwung",
        19: "befestigt.Sportanlage",
        20: "befestigt.andere_befestigte",
        21: "humusiert.Acker_Wiese_Weide",
        22: "humusiert.Reben",
        23: "humusiert.uebrige_Intensivkultur",
        24: "humusiert.Gartenanlage_Hausumschwung",
        25: "humusiert.Parkanlage",
        26: "humusiert.Sportanlage",
        27: "humusiert.Friedhof",
        28: "humusiert.Hoch_Flachmoor",
        29: "humusiert.Verkehrsteilerflaeche",
        31: "humusiert.andere_humusierte",
        32: "Gewaesser.stehendes",
        33: "Gewaesser.fliessendes",
        34: "Gewaesser.Schilfguertel",
        35: "bestockt.geschlossener_Wald",
        38: "bestockt.uebrige_bestockte",
        39: "vegetationslos.Fels",
        41: "vegetationslos.Geroell_Sand",
        43: "vegetationslos.Deponie",
    }

    # K values to test (realistic range for macro-classes)
    k_values = [6, 8, 10, 12, 15]

    # Process only cliped_22 for testing (smaller image)
    stack_path = stack_folder / "cliped_22_image_stack.tif"

    if not stack_path.exists():
        print(f"ERROR: {stack_path} not found!")
        exit(1)

    print(f"Processing: {stack_path.name}")

    # Calculate number of segments based on image size
    # Aim for ~10m² per superpixel (small enough for individual trees)
    with rasterio.open(stack_path) as src:
        area_m2 = src.width * src.res[0] * src.height * src.res[1]
        n_segments = int(area_m2 / 10)  # ~10m² per segment
        print(f"Image area: {area_m2/1e6:.2f} km², using {n_segments:,} segments (~10m² each)")

    # 1. Create superpixels (low compactness for edge-following)
    segments, metadata = create_superpixels(
        stack_path,
        n_segments=n_segments,
        compactness=1.0  # Low value = follows edges better
    )

    # 2. Extract features
    features = extract_superpixel_features(stack_path, segments)

    # 3. Cluster with multiple k values
    cluster_results = cluster_superpixels(features, k_values)

    # 4. Rasterize AV classes
    av_raster = rasterize_av_classes(
        gpkg_path,
        "Bodenbedeckung_BoFlaeche_Area",
        metadata
    )

    # 5. Analyze cluster composition for each k
    for k in k_values:
        cluster_labels = cluster_results[k]
        analyze_cluster_composition(
            segments,
            cluster_labels,
            av_raster,
            av_class_names,
            k
        )

        # Save cluster raster
        output_path = output_folder / f"{stack_path.stem}_clusters_k{k}.tif"
        save_cluster_raster(segments, cluster_labels, metadata, output_path, k)

    print("\n" + "="*60)
    print("Done! Check the cluster rasters in data/processed/clustering/")
    print("="*60)
