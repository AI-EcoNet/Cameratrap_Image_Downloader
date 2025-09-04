# 🦁 Animal Dataset Interactive Downloader

This small project aggregates **20+ camera trap datasets** from  
[https://lila.science/datasets](https://lila.science/datasets) and maps their metadata using the taxonomy mapping provided here:  
[https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/](https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/).

It provides access to **10+ million images** of animals, covering around **1000 different species** across multiple projects.

---

## ✨ Features

- 📂 Unified taxonomy across 20+ LILA datasets  
- 🐾 Explore ~1000 species interactively  
- 🔍 Select by **class → family → species** with live counts  
- ⬇️ Download images with:
  - Per‑species limits (e.g. max 2000 images per species)  
  - Automatic folder structure per species  
  - Metadata JSON for each species (traceable back to dataset & project)  
- ✅ Skips invalid/empty images and retries failed downloads  
- 📊 Provides statistics before downloading (counts, estimated size, etc.)  

---

## ⚠️ Notes on Data

- All **empty and non‑animal images** were removed.  
  - However, some bursts of images are grouped into one class, so not every image is guaranteed to contain the animal.  
- Some images may fail to download:
  - Certain links are marked as **private** and cannot be accessed.  
- Metadata is preserved so you can **cite the correct datasets/projects** when using the images.  

---

## 📦 Installation

Clone the repository and set up a virtual environment:

```bash
# Clone repo
git clone https://github.com/AI-EcoNet/Cameratrap_Image_Downloader.git
cd animal-dataset-downloader

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

All required files are included in the repo under `DataFiles/`.  
You can run the downloader immediately after installation:

```bash
python main.py
```

### Interactive Navigation

1. **Choose animal class** (e.g. Mammals, Birds, Reptiles, etc.)  
2. **Choose family/group** within the class  
3. **Choose species** (with consolidated names and counts)  
4. Confirm download options:
   - Download all  
   - Limit images per species  
   - Cancel and select more  

### Example Flow

🚀 ANIMAL DATASET INTERACTIVE DOWNLOADER


📂 Loading categories taxonomy...

⠋ Loading catalog (this may take a moment)

🦁 ANIMAL DATASET INTERACTIVE DOWNLOADER

🐾 Available animal classes:

1. Mammalia        (4,200,000 images)
2. Aves            (3,100,000 images)
3. Reptilia        (500,000 images)

...

---

## 📁 Output Structure

Downloads are stored under the `Downloads/` folder:

Downloads/

├── African Elephant (Loxodonta africana)/

│   ├── elephant_1234_uuid.jpg

│   ├── elephant_5678_uuid.jpg

│   └── African Elephant (Loxodonta africana).json

├── Red Fox (Vulpes vulpes)/

│   ├── fox_1234_uuid.jpg

│   └── Red Fox (Vulpes vulpes).json

└── failed_downloads.txt

Each species folder contains:
- ✅ Images  
- 📝 Metadata JSON with:
  - Species name  
  - Total images downloaded  
  - Source projects & datasets  
  - Download date  
  - Traceable image list  

---

## 📚 Future Work

- Upload a guide on how the `DataFiles/` were generated  
- Allow adding/removing/modifying taxonomy mappings  
- Improve handling of private/missing images  

---

## 🙋 Contributing

If you find this useful and would like to extend the taxonomy, add new datasets, or improve the downloader, feel free to open an issue or PR.  

---

## 📜 Citation

If you use this tool or the datasets, please **cite the original LILA datasets**:  
[https://lila.science/datasets](https://lila.science/datasets)

---

## 👋 Final Notes

This project is meant as a **practical helper** to quickly explore and download subsets of the massive LILA dataset collection.  
It is not a dataset itself, but a tool to make working with these datasets easier.

Enjoy exploring the animal kingdom! 🐆🦉🐍