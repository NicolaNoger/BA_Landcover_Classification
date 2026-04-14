import numpy as np
import os
import rasterio
import re # Neu: Für die numerische Sortierung

# Die Generator-Funktionen (snip_images_generator, snip_masks_generator) werden 
# im Hauptblock nicht mehr benötigt und können entfernt oder ignoriert werden. 
# Wir arbeiten direkt mit dem Hauptblock.

# --- Hauptskript mit Memory-Optimierung und korrekter Sortierung ---

if __name__ == "__main__":
    
    # Pfade definieren
    path = "/cfs/earth/scratch/nogernic/PA2/data/aerial/"
    img_path = "/cfs/earth/scratch/nogernic/PA2/data/aerial/img_tiles"
    mask_path = "/cfs/earth/scratch/nogernic/PA2/data/aerial/mask_tiles"
    
    # Zielordner erstellen
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)

    # 🔑 Funktion zur Extraktion der Zahl für die Sortierung 
    # (Damit "img10" nach "img9" kommt)
    def extract_number(f):
        # Sucht nach Ziffern im Dateinamen und gibt die gefundene Zahl als Integer zurück
        match = re.search(r'(\d+)', os.path.basename(f))
        return int(match.group(1)) if match else 0

    # Dateilisten erstellen und NUMERISCH sortieren!
    all_files = os.listdir(path)
    
    # Hier filtern wir nach deinen Namensmustern (z.B. "img" und "mask" am Anfang)
    img_files = sorted(
        [os.path.join(path, f) for f in all_files if f.endswith("image.tif")], 
        key=extract_number
    )
    mask_files = sorted(
        [os.path.join(path, f) for f in all_files if f.endswith("mask.tif")], 
        key=extract_number
    )
    
    if len(img_files) != len(mask_files):
        print("FEHLER: Die Anzahl der Bild- und Maskendateien stimmt nicht überein!")
        exit(1)

    print(f"Gefundene {len(img_files)} Bild-Masken-Paare.")
    
    # --- 1. Statistik sammeln (Speichereffizient) ---  
    print("\nStarte ersten Durchgang: Sammeln der globalen Normalisierungsstatistik...")
    

    total_sum = None
    total_sq_sum = None
    total_count = 0
    
    # Erster Durchgang: Statistik inkrementell über alle Snips sammeln
    for file in img_files:
        with rasterio.open(file) as src:
            image = src.read().transpose((1, 2, 0)).astype(np.float32)
            if total_sum is None:
                channel_count = image.shape[-1] 
                total_sum = np.zeros(channel_count, dtype=np.float64)
                total_sq_sum = np.zeros(channel_count, dtype=np.float64)

            img_height, img_width, _ = image.shape
            
            # Annahme snip_size=512, stride=256
            for y in range(0, img_height - 512 + 1, 256):
                for x in range(0, img_width - 512 + 1, 256):
                    snip = image[y:y + 512, x:x + 512]
                    
                    total_sum += snip.sum(axis=(0, 1))
                    total_sq_sum += (snip**2).sum(axis=(0, 1))
                    total_count += snip.shape[0] * snip.shape[1]

    mean = total_sum / total_count
    std = np.sqrt(total_sq_sum / total_count - mean**2)
    std = np.where(std == 0, 1.0, std).astype(np.float32)
    
    print(f"Global Mean: {mean}")
    print(f"Global Std: {std}")
    
    # --- 2. Zweiter Durchlauf: Generieren, Normalisieren und Speichern (Speichereffizient) ---
    print("\nStarte zweiten Durchgang: Generieren, Normalisieren und Speichern...")
    
    snip_index = 0
    # zip() paart die numerisch sortierten Listen korrekt!
    for img_file, mask_file in zip(img_files, mask_files):
        
        # Öffne Bild und Maske gleichzeitig, um sie paarweise zu verarbeiten
        with rasterio.open(img_file) as img_src, rasterio.open(mask_file) as mask_src:
            image = img_src.read().transpose((1, 2, 0)).astype(np.float32)
            mask = mask_src.read().transpose((1, 2, 0))
            
            img_height, img_width, _ = image.shape
            
            # Erzeuge Snips für Bild und Maske in identischer Reihenfolge
            for y in range(0, img_height - 512 + 1, 256):
                for x in range(0, img_width - 512 + 1, 256):
                    
                    img_snip = image[y:y + 512, x:x + 512]
                    mask_snip = mask[y:y + 512, x:x + 512]
                    
                    # Normalisieren und sofort speichern (speichereffizient)
                    normalized_snip = (img_snip - mean) / std 
                    
                    img_save_path = os.path.join(img_path, f"img_{snip_index:06d}.npy")
                    mask_save_path = os.path.join(mask_path, f"mask_{snip_index:06d}.npy")
                    
                    np.save(img_save_path, normalized_snip)
                    np.save(mask_save_path, mask_snip)
                    
                    snip_index += 1

    print(f"\nFertig! Es wurden insgesamt {snip_index} Snips gespeichert.")