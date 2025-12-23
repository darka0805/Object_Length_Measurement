# Object Length Measurement

This project provides an automated computer vision pipeline to detect, segment, and measure the length of *C. elegans* (or similar microscopic worms) from images. It is designed to be robust against noise, low contrast, and background artifacts commonly found in microscopy data.

## Features

* **Advanced Preprocessing:** Uses **CLAHE** and **Bilateral Filtering** to enhance worm visibility while suppressing noise.
* **Robust Segmentation:** Combines Otsu's thresholding with geometric filtering to distinguish worms from background debris.
* **Automatic Repair:** Uses morphological "gluing" to reconnect worm segments that may have been broken during thresholding.
* **Precise Measurement:**
    * **Skeletonization:** Reduces the worm to a 1-pixel wide centerline.
    * **Graph Theory:** Uses `networkx` to find the longest geodesic path along the skeleton.
    * **Spline Smoothing:** Applies B-Spline interpolation for sub-pixel accuracy.

## Dependencies

This project requires the following libraries:

```bash
pip install opencv-python numpy networkx matplotlib scikit-image scipy
```
## Project Structure
.
├── main.py                
├── data/                  
├── output_results/        
└── README.md              

## Pipeline
Enhancement: The image is converted to grayscale and contrast is boosted using CLAHE. A Gaussian blur is applied to smooth out grain, followed by a Bilateral Filter to preserve edges.

Segmentation: The image is thresholded. Contours are analyzed; small or non-worm-like objects are discarded.

Glue & Clean: A morphological closing operation connects nearby components to ensure the worm is a single continuous object.

Skeletonization: The binary shape is thinned to a skeleton.

Pathfinding: The skeleton is converted into a graph. The longest path between any two endpoints is calculated to determine the length.

## Results & Conclusions

The implemented pipeline relies on **classical computer vision techniques** (Contrast Enhancement, Adaptive Thresholding, and Morphological "Gluing").

The majority of the dataset was processed successfully. The algorithm is highly effective at detecting worms with stable intensity, even when they are significantly curled or surrounded by small debris.

### Key Observations from the Dataset:

- The pipeline reliably isolates the worm body while ignoring small debris and noisy background textures.
- Graph-based pathfinding accurately tracks the centerline of highly curved and U-shaped worms without cutting corners.
- The gluing step reconnects fragmented masks in low-contrast images, enabling continuous and reliable measurements.


### Limitations & Future Improvements

Hovewer, in cases like `result_64`, where the worm is physically touching or overlapping with large, dark dirt clumps, the intensity-based segmentation merges the two objects. The algorithm then attempts to skeletonize the entire blob, leading to incorrect measurements. For future usage of deep learning segmantation would make the results much better
