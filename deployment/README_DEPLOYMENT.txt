NPR Detector Deployment Package

Files:
1. best_npr_detector.pt
   - Trained PyTorch checkpoint.

2. model_metadata.json
   - Contains input size, normalization, class mapping, NPR transform info, and training/evaluation summary.

3. inference.py
   - Contains the model architecture and prediction wrapper.

4. requirements.txt
   - Python packages needed for deployment.

How to use locally:

python inference.py --package_dir ./npr_detector_deployment_package --image path/to/image.jpg

Expected output:
{
  "pred_label": 1,
  "pred_name": "fake",
  "prob_real": 0.01,
  "prob_fake": 0.99
}

Class mapping:
0 = real
1 = fake

Important:
The model architecture inside inference.py must stay unchanged because it must match the saved checkpoint.
