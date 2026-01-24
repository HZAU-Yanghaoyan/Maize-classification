
## 1. System Requirements

### Software Dependencies

* **MATLAB** (Version R2021a)
* **Operating System** (Tested):
  * Windows 10 (64-bit)

### Hardware Requirements

* Recommended configuration (for training):
  * 16 GB RAM or higher
  * NVIDIA GPU
  * 10 GB of available disk space

## 2. Installation Guide

### Installation Steps

1. Clone the repository to your local machine:

   ```bash
   git clone [repository-url]
   ```

2. Make sure MATLAB is installed.

3. Extract the demo dataset:

   * Unzip the `demo.rar` file
   * The extracted directory should contain sample image files

### Typical Installation Time

* On a standard desktop computer: approximately 5–10 minutes

## 3. Demo

### Running the Demo

1. Open MATLAB
2. Set the current directory to the folder containing the code
3. Run the demo script:

   ```matlab
   Image_Classification_OurModel_demo
   ```

### Expected Output

* Model architecture visualization
* Image classification accuracy
* Summary table of deep phenotypes

### Expected Run Time

* On a standard desktop computer: approximately 5–10 minutes

## 4. Usage Instructions

### Running the Software on Your Own Data

1. **Data Preparation**:

   * Supported formats: `.jpg`, `.png`
   * Image size: recommended 224 × 224 pixels (automatic resizing supported)
   * Color space: RGB
   * Data organization: one subfolder per class

2. **Modify Configuration**:

   * Edit the `Location` variable in the script to point to your data directory

3. **Run the Model**:

   ```matlab
   % Train OurModel
   Image_Classification_OurModel

   % Or use ResNet18 as a baseline
   Image_Classification_ResNet18
   ```

4. **Parameter Tuning**:

   * Training parameters can be adjusted in the `options` section:

     * Learning rate
     * Batch size
     * Number of epochs
     * Optimizer settings

## 5. File Description

### Main Files

* `Image_Classification_OurModel.m` – Main implementation of our proposed model
* `Image_Classification_OurModel_demo.m` – Demo script
* `Image_Classification_ResNet18.m` – Baseline models such as ResNet18
* `demo.rar` – Demo dataset (contains sample images)

## Citation

If you use this code in your research, please cite our paper.

## Support

If you have any questions, please submit an issue via GitHub Issues or contact the authors.

---

**Note**: This code is provided for academic research purposes only. Commercial use may require additional licensing.
