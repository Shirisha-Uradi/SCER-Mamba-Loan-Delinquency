import os
import shutil

folders = [
    r"Data/sequential_enhanced_model_ready_age18",
    r"Data/sequential_enhanced_model_ready_age24"
]

for folder in folders:

    if not os.path.exists(folder):
        print("NOT FOUND:", folder)
        continue

    zip_path = folder + ".zip"

    shutil.make_archive(
        folder,
        "zip",
        folder
    )

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)

    print(
        f"Created: {zip_path} | "
        f"Size: {size_mb:.2f} MB"
    )

print("\nJ18 + J24 ZIP COMPLETE")