# Product Visual Search

This application retrieves products from the Amazon Berkeley Objects (ABO) catalogue (https://amazon-berkeley-objects.s3.amazonaws.com/index.html) based on an input image provided by the user. 

The ABO dataset contains approximately 147,702 product listings and 398,212 catalogue images. The product listings are provided in JSONL format, while image metadata is available in a compressed CSV file. 
These are processed and combined to create a unified product catalogue linking product information, image IDs, and image file paths. 
It's built with OpenCLIP (CLIP ViT-B/32), a pretrained vision transformer that preprocesses each image in the catalogue and encodes it as a 512 dimensional vector to create embeddings. Each image is 
preprocessed using the model's standard transformations and passed through the encoder to obtain visual features of that image. These feature embeddings are stored as NumPy arrays and indexed using FAISS (IndexFlatIP) which conducts a similarity search between products. 

When a user uploads an image it undergoes the same preprocessing and encoding to produce an embedding, which is queried against the FAISS index to retrieve the most similar product in the catalogue. The retrieved
image IDs are mapped back to the product catalogue and results are gouped by product ID so we don't get multiple entries for the same product. The highest-scoring image is retained for each product and the
top matches are displayed.

### Run the Application

Install required dependencies:

`pip install -r requirements.txt`

Set the dataset and artifact directories using the environment variables DATASET_DIR and ARTIFACT_DIR, then launch the application:

`streamlit run app.py`

### Docker

Build the Docker image:

`docker build -t abo-visual-search .`

Run the container:

`docker run -p 8501:8501 -v /path/to/dataset:/data/project:ro -v /path/to/artifacts:/data/artifacts abo-visual-search`

The application will be available at:

http://localhost:8501

The dataset and precomputed artifacts are mounted separately rather than bundled into the Docker image.

Since this is only a demo, recurring costs were kept in mind and the app is hosted through a Cloudflare Tunnel running on the host machine instead of a paid VM or server. The tunnel requires an active connection from the host so, please contact me to coordinate the Cloudflare Tunnel setup and activation before accessing the application. I would be happy to coordinate a time for a free demo and conversation!
