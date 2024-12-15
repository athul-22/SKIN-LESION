import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

class MelanomaModel(torch.nn.Module):
    def __init__(self, out_size, dropout_prob=0.5):
        super(MelanomaModel, self).__init__()
        from efficientnet_pytorch import EfficientNet
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficient_net._fc = torch.nn.Identity()
        self.fc1 = torch.nn.Linear(1280, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, out_size)
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.efficient_net(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Diagnosis mapping
DIAGNOSIS_MAP = {
    0: 'Melanoma',
    1: 'Melanocytic nevus',
    2: 'Basal cell carcinoma',
    3: 'Actinic keratosis',
    4: 'Benign keratosis',
    5: 'Dermatofibroma',
    6: 'Vascular lesion',
    7: 'Squamous cell carcinoma',
    8: 'Unknown'
}

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MelanomaModel(out_size=9)
model.load_state_dict(torch.load('../model/multi_weight.pth', map_location=device)['model_state_dict'])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Open and transform the image
        image = Image.open(filepath)
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(outputs, 1)
            
            # Get top 3 predictions
            top_3_probs, top_3_indices = torch.topk(probabilities, 3)
            
            predictions = []
            for prob, idx in zip(top_3_probs[0], top_3_indices[0]):
                predictions.append({
                    'diagnosis': DIAGNOSIS_MAP[idx.item()],
                    'probability': prob.item()
                })
            
            return jsonify({
                'predictions': predictions
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)