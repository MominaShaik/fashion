# Intelligent Fashion Product Catalog Visualizer

An automated deep-learning and analytics ecosystem engineered to manage large-scale fashion product inventories. The system ingests categorical relational data, cleans missing or incomplete records, handles outliers, provides high-dimensional exploratory data visualization, and builds scalable data architectures to automate visual product categorization and duplicate mitigation.

## 🚀 Key Features

- **Robust Data Pipeline:** Automatically handles multi-column skips on bad lines and executes strategic drop-row processes to eliminate structural missingness (`NaN`).
- **Dynamic Sampling & Visual Analytics:** A dynamic, parameter-driven statistical visualizer built on `matplotlib` that samples structural attributes like `gender`, `masterCategory`, `subCategory`, and `articleType` to plot and preview real-time asset tracking grids.
- **Deep Feature Representation Pipeline:** Incorporates deep convolutional structures via `TensorFlow` to process image matrices, extracting low-level feature embeddings and transforming raw pixel counts into distinct feature matrices.
- **Catalog Clustering & Deduplication System:** An embedded programmatic grouping model designed to handle automated item grouping and redundant item compression. It resolves multi-row identical groups into single, gold-standard master items to clean cluttered upstream database states.

---

## 📂 Dataset Architecture

The project operates seamlessly against the **Fashion Product Images (Small)** repository, matching tabular data streams directly with discrete structural images.

```
/kaggle/input/fashion-product-images-small/
│
├── styles.csv         # Core metadata (id, gender, category, subCategory, articleType, etc.)
└── images/            # Directory containing JPEG image vectors matched by unique id keys
```

### Relational Schema Sample

The tabular data contains the following structural features per entry:
* **id**: Unique asset identifier matching the corresponding filename index `[id].jpg`.
* **gender**: Target gender classification segment (`Men`, `Women`, `Boys`, etc.).
* **masterCategory**: Macro-level department indices (`Apparel`, `Accessories`, `Footwear`, `Personal Care`).
* **subCategory**: Structural class indicators (`Topwear`, `Bottomwear`, `Watches`, etc.).
* **articleType**: Micro-level merchandise classifications (`Shirts`, `Jeans`, `Tshirts`, etc.).
* **baseColour**, **season**, **year**, **usage**, **productDisplayName**: Textual identifiers and visual context fields.

---

## 🛠️ Technological Stack

* **Framework**: Jupyter Notebook / Python 3 Environment
* **Core Vector Operations**: `numpy`
* **Data Structuring / Pipelines**: `pandas`
* **Mathematical Plotting & Render Engines**: `matplotlib`
* **Deep Learning Engine**: `TensorFlow` / `Keras` (leveraging Nvidia Tesla T4 GPU accelerators for processing feature maps)

---

## ⚙️ Setup and Usage Instructions

* **Environment Configuration:** Ensure you are executing inside a Python 3 framework with GPU support enabled if processing massive iterations of item embeddings.
* **Dependencies:** Install all critical runtime components via terminal piping:
  ```bash
  pip install numpy pandas matplotlib tensorflow
  ```
* **Data Mounting:** Download the fashion-product-images-small files and update your image_path parameters to reflect local environment paths.
* **Run Pipeline:** Execute the notebook sequentially to load metadata, render visualization sheets, map data structures, and trigger inventory deduplication checks.
