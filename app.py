
import streamlit as st
import torch
import open_clip
import faiss
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

import os

DATASET_DIR = Path(
    os.getenv(
        "DATASET_DIR",
        r"D:\PixoAnalytics\Image Retrieval Project"
    )
)

IMAGE_DIR = DATASET_DIR / "images" / "small"

ARTIFACT_DIR = Path(
    os.getenv(
        "ARTIFACT_DIR",
        r"D:\PixoAnalytics\Image Retrieval Artifacts"
    )
)

EMBEDDING_DIR = ARTIFACT_DIR / "embeddings"

INDEX_PATH = EMBEDDING_DIR / "faiss_index.bin"
IDS_PATH = EMBEDDING_DIR / "image_ids.npy"
POSITIONS_PATH = EMBEDDING_DIR / "valid_positions.npy"
CATALOGUE_PATH = ARTIFACT_DIR / "catalogue.parquet"


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Image Retrieval Product Search",
    layout="wide"
)

st.title("Visual Product Search")
st.caption(
    "Upload an image and find similar products in the Amazon Berkeley Objects catalogue"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        device=device
    )

    model.eval()

    return model, preprocess, device


# --------------------------------------------------
# LOAD INDEX AND DATA
# --------------------------------------------------

@st.cache_resource
def load_index():

    index = faiss.read_index(str(INDEX_PATH))

    image_ids = np.load(
        IDS_PATH,
        allow_pickle=False
    )

    valid_positions = np.load(
        POSITIONS_PATH,
        allow_pickle=False
    )

    return index, image_ids, valid_positions


@st.cache_data
def load_catalogue():

    catalogue = pd.read_parquet(CATALOGUE_PATH)

    return (
        catalogue[
            [
                "image_id",
                "product_id",
                "path",
                "title",
                "brand",
                "product_type",
                "color",
                "style"
            ]
        ]
        .drop_duplicates(subset=["image_id", "product_id"])
        .reset_index(drop=True)
    )


# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

def retrieve_products(
    query_image,
    model,
    preprocess,
    device,
    index,
    image_ids,
    valid_positions,
    catalogue,
    top_k=10
):

    query = preprocess(
        query_image.convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.inference_mode():

        embedding = model.encode_image(query)

        embedding = (
            embedding /
            embedding.norm(dim=-1, keepdim=True)
        )

    query_vector = np.ascontiguousarray(
        embedding.cpu().numpy().astype(np.float32)
    )

    # Retrieve a larger image candidate pool,
    # then rank unique products.
    candidate_k = min(500, index.ntotal)

    scores, positions = index.search(
        query_vector,
        candidate_k
    )

    original_positions = valid_positions[positions[0]]

    matched_ids = image_ids[original_positions]

    matches = pd.DataFrame({
        "image_id": matched_ids,
        "similarity": scores[0]
    })

    results = matches.merge(
        catalogue,
        on="image_id",
        how="inner"
    )

    # Retain the highest-scoring image per product
    results = (
        results
        .sort_values("similarity", ascending=False)
        .drop_duplicates(subset="product_id")
        .head(top_k)
        .reset_index(drop=True)
    )

    return results


# --------------------------------------------------
# INITIALIZE SYSTEM
# --------------------------------------------------

try:

    with st.spinner("Loading retrieval system..."):

        model, preprocess, device = load_model()

        index, image_ids, valid_positions = load_index()

        catalogue = load_catalogue()

    st.sidebar.success("Retrieval system ready")

    st.sidebar.write(f"Device: {device}")
    st.sidebar.write(f"Indexed images: {index.ntotal:,}")
    st.sidebar.write(f"Products: {catalogue['product_id'].nunique():,}")

except Exception as e:

    st.error(f"Could not initialize retrieval system: {e}")
    st.stop()


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.subheader("Upload a product image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    query_image = Image.open(uploaded_file).convert("RGB")

    st.image(
        query_image,
        caption="Your query image",
        width=300
    )

    if st.button("Find similar products", type="primary"):

        with st.spinner("Searching the catalogue..."):

            results = retrieve_products(
                query_image,
                model,
                preprocess,
                device,
                index,
                image_ids,
                valid_positions,
                catalogue,
                top_k=10
            )

        st.subheader("Similar products")

        if results.empty:

            st.warning("No matching products found.")

        else:

            columns = st.columns(5)

            for i, (_, row) in enumerate(results.iterrows()):

                image_path = IMAGE_DIR / row["path"]

                with columns[i % 5]:

                    if image_path.exists():
                        st.image(
                            str(image_path),
                            use_container_width=True
                        )

                    st.markdown(
                        f"**{str(row['title'])[:80]}**"
                    )

                    st.caption(
                        f"Brand: {row['brand']}"
                    )

                    st.caption(
                        f"Category: {row['product_type']}"
                    )

                    st.caption(
                        f"Similarity: {row['similarity']:.4f}"
                    )

                    st.caption(
                        f"Product ID: {row['product_id']}"
                    )
