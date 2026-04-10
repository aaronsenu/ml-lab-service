# Lab 2 – Secrets Injection, CI/CD & House Segmentation

## Project Structure

```
lab2/
├── app.py                          # Flask API (Lab 1 + secrets injection)
├── generate_masks.py               # Week 7 pixel mask generation
├── train_segmentation.py           # UNet training & evaluation
├── test_app.py                     # Unit tests for API
├── Dockerfile                      # Container definition
├── requirements.txt                # All dependencies
├── .env.example                    # Template for secrets (commit this)
├── .env                            # Actual secrets  ← DO NOT COMMIT
├── .gitignore
└── .github/
    └── workflows/
        └── cicd.yml                # GitHub Actions CI/CD pipeline
```

---

## 1 – Secrets Injection

Sensitive values (API key, Docker Hub credentials) are stored in `.env` and
loaded at runtime via `python-dotenv`. They are **never hard-coded**.

```bash
cp .env.example .env
# Edit .env and fill in your values
```

Run the Flask app locally:
```bash
pip install -r requirements.txt
python app.py
```

Test with the API key header:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_api_key_here" \
  -d '{"text": "This is amazing!"}'
```

Health check (no key required):
```bash
curl http://localhost:5000/health
```

---

## 2 – Run with Docker

```bash
# Build
docker build -t ml-lab-service:latest .

# Run (secrets injected via --env-file, NOT baked into image)
docker run --env-file .env -p 5000:5000 ml-lab-service:latest
```

Docker Hub image: `aar0ns3nu/ml-lab-service:latest`

---

## 3 – CI/CD (GitHub Actions)

Pipeline stages (`.github/workflows/cicd.yml`):
1. **Test** – runs `pytest test_app.py`
2. **Build & Push** – builds Docker image and pushes to Docker Hub (only on passing tests)

### Required GitHub Secrets
Go to your repo → **Settings → Secrets → Actions → New repository secret**:

| Secret name           | Value                        |
|-----------------------|------------------------------|
| `API_KEY`             | Your API key                 |
| `DOCKER_HUB_USERNAME` | `aar0ns3nu`                  |
| `DOCKER_HUB_TOKEN`    | Docker Hub access token      |

---

## 4 – Dataset Preparation (Pixel Mask Generation)

```bash
# Generate synthetic demo dataset (120 samples, split 70/15/15)
python generate_masks.py --demo --out_dir data --n_samples 120

# Or process a real dataset (Inria layout):
python generate_masks.py --images_dir raw/images --masks_dir raw/masks --out_dir data
```

---

## 5 – Train Segmentation Model

```bash
python train_segmentation.py \
    --data_dir data \
    --out_dir  outputs \
    --epochs   15 \
    --batch_size 8
```

**Model**: UNet with ResNet34 encoder (pretrained on ImageNet)  
**Loss**: Dice Loss  
**Metrics reported**: IoU, Dice Score  

Outputs saved to `outputs/`:
- `best_model.pth`        – best checkpoint
- `metrics.json`          – final test IoU & Dice
- `training_curves.png`   – loss / IoU / Dice curves
- `predictions.png`       – sample aerial / GT mask / predicted mask

---

## 6 – Run Unit Tests

```bash
pytest test_app.py -v
```
