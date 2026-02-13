# Sign Language Translation (SLT) API

Advanced Sign Language Recognition and Translation system using Transformer-based Seq2Seq architecture with MediaPipe landmarks and I3D features.

## 🎯 Project Overview

This FastAPI application provides real-time sign language video translation to English text using:
- **Transformer Seq2Seq Model** (4 encoder + 4 decoder layers, 512 d_model)
- **MediaPipe Holistic** for 1,629 spatial keypoints (pose, face, hands)
- **I3D Features** for 1,024 temporal features
- **Advanced Beam Search** with n-gram blocking, temperature decay, and linguistic filtering

**Target Accuracy**: 90-95% | **WER Target**: < 10%

## 📋 Features

### Core Capabilities
- ✅ Real-time video processing (200 frames @ 2653 features/frame)
- ✅ Advanced decoding with beam search (width=3)
- ✅ Temperature decay (0.8 → 0.4) for confidence progression
- ✅ English bigram filtering for linguistic plausibility
- ✅ Feature balancing (MediaPipe 70%, I3D 30%)
- ✅ Global Z-score normalization
- ✅ Grammar correction with TextBlob
- ✅ Cross-attention validation
- ✅ CORS-enabled REST API

### Technical Optimizations
- **Input Normalization**: LayerNorm + Global StandardScaler
- **Sampling**: Top-K (50) + Top-P (0.95) + Temperature Decay
- **Repetition Control**: 3-gram blocking + biased token penalties
- **Memory Validation**: Encoder output variance checking
- **Linguistic Filtering**: English bigram boost (3x)

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository**:
```bash
cd "d:\All Projects\ASLR"
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download required files**:
   - `sign_language_FINAL_A100_SUCCESS.pth` (165 MB) - Trained model
   - `vocab.pkl` (205 KB) - Vocabulary (10,160 tokens)
   - `rgb_imagenet.pt` (50 MB) - I3D pretrained weights

4. **Verify installation**:
```bash
python test_model_load.py
```

## 🎬 Running the Application

### Development Mode
```bash
python main.py
```
Or:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Batch Scripts
**Windows**:
```bash
start_server.bat
```

**Linux/Mac**:
```bash
chmod +x start_server.sh
./start_server.sh
```

API will be available at: `http://localhost:8000`

## 📡 API Endpoints

### Health Check
```http
GET /
```
**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "vocab_size": 10160
}
```

### Prediction
```http
POST /predict
Content-Type: multipart/form-data
```

**Request**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sign_video.mp4"
```

**Response**:
```json
{
  "prediction": "I want to buy a book"
}
```

**Supported Formats**: MP4, AVI, MOV, MKV, WMV

## 🔧 Configuration

### Model Architecture
```python
input_dim = 2653        # MediaPipe (1629) + I3D (1024)
d_model = 512           # Transformer hidden size
nhead = 8               # Attention heads
num_encoder_layers = 4
num_decoder_layers = 4
vocab_size = 10160
max_length = 20         # Output length
```

### Decoding Parameters
```python
beam_width = 3
temperature_start = 0.8  # First 3 words
temperature_end = 0.4    # Rest of sentence
top_k = 50
top_p = 0.95
ngram_blocking = 3
length_penalty = 0.6
```

### Feature Weighting
```python
mediapipe_weight = 0.7
i3d_weight = 0.3
```

## 🧪 Testing

### Test Model Loading
```bash
python test_model_load.py
```

### Test Prediction
```bash
python test_prediction.py
```

### Test API
```bash
python test_api.py
```

### Test with Video
```bash
python video_test.py
```

## 📊 Technical Architecture

### Pipeline Flow
```
Video Input (MP4)
    ↓
MediaPipe Holistic (1629 landmarks)
    ↓
I3D Feature Extraction (1024 features)
    ↓
Feature Weighting (0.7 + 0.3)
    ↓
Global Z-Score Normalization
    ↓
Transformer Encoder (4 layers)
    ↓
Memory Validation
    ↓
Transformer Decoder (4 layers)
    ↓
Beam Search + Temperature Decay
    ↓
Bigram Filtering + Grammar Correction
    ↓
English Text Output
```

### Key Components

**`main.py`**: FastAPI server + feature extraction
**`model.py`**: Transformer architecture with LayerNorm
**`advanced_predict_translation.py`**: Beam search + linguistic filtering
**`i3d.py`**: I3D feature extractor
**`vocab.pkl`**: Tokenizer vocabulary

## 🎯 Performance Optimizations

### Feature Engineering
1. **Balanced Weighting**: Prevents I3D from overpowering landmarks
2. **Global Normalization**: Aligns with training distribution
3. **LayerNorm**: Stabilizes input projection

### Decoding Intelligence
1. **Temperature Decay**: Exploration → Confidence
2. **Bigram Filtering**: Boosts valid word pairs ("I want", "want to")
3. **Biased Token Penalty**: 95% penalty on repetitive words
4. **Grammar Correction**: TextBlob post-processing

### Memory Efficiency
- Automatic temp file cleanup
- Efficient tensor operations
- Batch processing support

## 🐛 Troubleshooting

### Common Issues

**1. Low Confidence (~4%)**
- ✅ Fixed with temperature decay and bigram filtering

**2. Word Soup Output**
- ✅ Fixed with linguistic filtering and grammar correction

**3. Identical Outputs for Different Videos**
- ✅ Fixed with memory validation and feature debugging

**4. Model Loading Error**
```bash
# Verify files exist
ls -lh sign_language_FINAL_A100_SUCCESS.pth vocab.pkl
```

**5. CUDA Out of Memory**
```python
# Reduce batch size or use CPU
device = torch.device('cpu')
```

## 📝 Debug Logs

When running, you'll see:
```
I3D Features - Mean: X, Std: Y, Min: Z, Max: W
RAW Features - Mean: X, Std: Y
Global Normalization - Mean: X, Std: Y
NORMALIZED Features - Mean: X, Std: Y
Memory stats - Mean: X, Std: Y
DEBUG: Step 0, Temperature: 0.8
DEBUG: Boosted bigram 'i' -> 'want' (ID 234)
DEBUG: Penalized biased token 1729
```

## 🔬 Advanced Features

### English Bigram Dictionary
```python
'i': ['am', 'want', 'need', 'have', 'can']
'want': ['to', 'the', 'a']
'to': ['buy', 'go', 'see', 'get']
```

### Biased Token Penalties
```python
# 95% penalty on repetitive tokens
biased_tokens = [1729, 187, 2341, 1456]  # pour, push, was, silk
```

## 📚 Project Structure

```
ASLR/
├── main.py                          # FastAPI server
├── model.py                         # Transformer architecture
├── advanced_predict_translation.py  # Beam search + filtering
├── i3d.py                          # I3D feature extractor
├── vocab.pkl                       # Vocabulary
├── sign_language_FINAL_A100_SUCCESS.pth  # Trained model
├── rgb_imagenet.pt                 # I3D weights
├── requirements.txt                # Dependencies
├── config.json                     # Configuration
├── test_*.py                       # Test scripts
└── start_server.*                  # Launch scripts
```

## 🤝 Integration Example (C#)

```csharp
using System.Net.Http;
using System.IO;

var client = new HttpClient();
var form = new MultipartFormDataContent();
var fileContent = new StreamContent(File.OpenRead("video.mp4"));
form.Add(fileContent, "file", "video.mp4");

var response = await client.PostAsync(
    "http://localhost:8000/predict", 
    form
);
var result = await response.Content.ReadAsStringAsync();
Console.WriteLine(result);  // {"prediction": "I want to buy..."}
```

## 📄 License

This project is part of a 100,000 PKR Sign Language Translation system.

## 👨‍💻 Author

Developed with advanced Transformer architecture and linguistic optimization techniques.

---

**Last Updated**: February 2026
**Version**: 2.0 (Feature Scaling & Confidence Optimization)