# NPR_CV
NPR Deepfake Detection to detect between real and fake images replicating this paper's model"https://arxiv.org/abs/2312.10461"

Project: NPR Deepfake Detection
Goal: Detect whether an input image is real or fake/generated.
Input: RGB image.
Output: real/fake prediction with probability.
Approach: NPR transform + lightweight CNN detector.
Dataset: ForenSynths/Wang_CVPR2020.
Main result: 91.10% fair Table 1-style mean accuracy vs paper 92.45%.
Deployment: Flask/Docker REST API.

Main commands:

pip install -r requirements.txt
python src/inference.py --image path/to/image.jpg --model models/best_npr_detector.pt

For deployment:

cd deployment
docker build -t npr-detector .
docker run -p 5000:5000 npr-detector
