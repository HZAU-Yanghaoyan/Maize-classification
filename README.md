## 1. System Requirements

### Software Dependencies

* **MATLAB** (Version R2021a or newer recommended)
* **Operating System** (Tested):
  * Windows 10 (64-bit)

### Hardware Requirements

* Recommended configuration (for training):
  * 16 GB RAM or higher
  * NVIDIA GPU (with CUDA support)
  * 10 GB of available disk space

## 2. Installation Guide

### Installation Steps

1. Clone the repository to your local machine:

   ```bash
   git clone [repository-url]
   ```

2. Ensure MATLAB is installed with the required toolboxes:
   * Deep Learning Toolbox
   * Image Processing Toolbox
   * Computer Vision Toolbox (optional, for advanced preprocessing)

3. Prepare your dataset according to the instructions in [Usage Instructions](#4-usage-instructions).



## 3. Usage Instructions

### Data Preparation

* **Supported formats**: `.jpg`, `.png`, `.bmp`
* **Image size**: The model automatically resizes images to 224×224 pixels; no manual resizing required.
* **Color space**: RGB images only.
* **Folder structure**: Place images in subfolders, one subfolder per class. For example:

  ```
  your_dataset/
      class_1/
          image1.jpg
          image2.jpg
          ...
      class_2/
          image1.jpg
          ...
  ```

### Running the Model

1. Open the script `Image_Classification_OurModel.m` (or `Image_Classification_ResNet18.m`) in MATLAB.
2. Locate the line that sets the `Location` variable (near the top of the script) and change it to the path of your dataset folder.
3. Run the script. The model will:
   * Load and split the data into training and validation sets.
   * Train the network.
   * Display training progress, accuracy, and a classification summary.

### Parameter Tuning

You can modify training options inside the script:

* **Learning rate**
* **Mini-batch size**
* **Max epochs**
* **Optimizer** (e.g., `sgdm`, `adam`)

These options are defined using the `trainingOptions` function in MATLAB.

## 4. File Description

### Main Files

* `Image_Classification_OurModel.m` – Main implementation of our proposed deep learning model for image classification.
* `Image_Classification_ResNet18.m` – Baseline implementation using the pre-trained ResNet18 network (for comparison).


## Citation

If you use this code in your research, please cite our paper:

```
[Paper citation details to be added]
```

## Support

For questions or issues, please open an issue on GitHub or contact the authors directly.

---

**Note**: This code is provided for academic research purposes only. Commercial use may require additional licensing.
