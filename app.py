import os
import io
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import open_clip

app = Flask(__name__)

# Initialize the OpenCLIP model
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED_WEIGHTS = "openai"

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_WEIGHTS)
model.eval()

tokenizer = open_clip.get_tokenizer(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load precomputed image embeddings
embedding_df = pd.read_pickle("image_embeddings.pickle")
image_embeddings = np.vstack(embedding_df['embedding'].values)
image_filenames = embedding_df['file_name'].tolist()

def extract_text_embedding(query):
    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        normalized = F.normalize(embedding, p=2, dim=1)
    return normalized.cpu().numpy()

def extract_image_embedding(image):
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        normalized = F.normalize(embedding, p=2, dim=1)
    return normalized.cpu().numpy()

def reduce_dimensions(data, components):
    pca = PCA(n_components=components)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca

@app.route('/', methods=['GET', 'POST'])
def search():
    search_results = []

    if request.method == 'POST':
        text_input = request.form.get('text_query', '').strip()
        weight = request.form.get('lam', '0.5').strip()
        k_pca = request.form.get('pca_k', '').strip()

        try:
            weight = float(weight) if weight else 0.5
        except ValueError:
            weight = 0.5  # Default weight

        image_file = request.files.get('image_query')
        has_text = bool(text_input)
        has_image = image_file and image_file.filename

        # Determine PCA usage
        apply_pca = False
        if k_pca:
            try:
                k_pca = int(k_pca)
                image_emb = image_embeddings.copy()
                image_emb, pca_model = reduce_dimensions(image_embeddings, k_pca)
                apply_pca = True
            except ValueError:
                apply_pca = False
        else:
            image_emb = image_embeddings

        # Initialize query vector
        query_vector = None

        if has_text and not has_image:
            # Handle text-only query
            text_emb = extract_text_embedding(text_input)
            if apply_pca:
                text_emb = pca_model.transform(text_emb)
            query_vector = text_emb

        elif has_image and not has_text:
            # Handle image-only query
            try:
                img = Image.open(io.BytesIO(image_file.read())).convert("RGB")
                img_emb = extract_image_embedding(img)
                if apply_pca:
                    img_emb = pca_model.transform(img_emb)
                query_vector = img_emb
            except Exception:
                pass  

        elif has_text and has_image:
            # Handle combined text and image query
            try:
                img = Image.open(io.BytesIO(image_file.read())).convert("RGB")
                img_emb = extract_image_embedding(img)
                text_emb = extract_text_embedding(text_input)

                if apply_pca:
                    img_emb = pca_model.transform(img_emb)
                    text_emb = pca_model.transform(text_emb)

                query_vector = (weight * text_emb) + ((1 - weight) * img_emb)
            except Exception:
                pass 

        # If a query vector is available, compute similarities
        if query_vector is not None:
            sim_scores = cosine_similarity(query_vector, image_emb)[0]
            top_indices = np.argsort(sim_scores)[-5:][::-1]  # Top 5 indices

            for idx in top_indices:
                search_results.append({
                    "file_name": image_filenames[idx],
                    "similarity": float(sim_scores[idx])
                })

    return render_template('index.html', results=search_results)

if __name__ == '__main__':
    app.run(debug=True)
