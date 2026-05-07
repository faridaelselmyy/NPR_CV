# NPR Deepfake Detection

This project implements Neighboring Pixel Relationships (NPR) for generalizable fake/generated image detection.

## Goal
Input: RGB image  
Output: Real/Fake prediction with probabilities.

## Method
The image is converted into an NPR representation using 2x2 neighboring pixel differences, then classified using a lightweight CNN detector.

## Dataset
ForenSynths / Wang_CVPR2020.

## Main Results
- Phase 2 validation accuracy: 99.94%
- Phase 3 overall accuracy over 13 sources: 79.34%
- Fair Table 1-style accuracy over 8 paper-matching sources: 91.10%
- Paper Table 1 mean accuracy: 92.45%

## Deployment
See `deployment/README_DEPLOYMENT.txt`.

## Project Structure
- `notebooks/`: preprocessing, training, evaluation
- `deployment/`: model checkpoint and inference code
- `results/`: metrics and CSV files
- `report/`: final PDF documentation
