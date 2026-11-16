# Product Visual Search

This application retrieves products from the Amazon Berkeley Objects (ABO) catalogue (https://amazon-berkeley-objects.s3.amazonaws.com/index.html) based on an input image provided by the user. 
The ABO dataset contains approximately 147,702 product listings and 398,212 catalogue images. The product listings are provided in JSONL format, while image metadata is available in a compressed CSV file. 
These are processed and combined to create a unified product catalogue linking product information, image IDs, and image file paths. 
It's built with OpenCLIP (CLIP ViT-B/32), a pretrained vision transformer 
that preprocesses each image in the catalogue and encodes it as a 512 dimensional vector to create embeddings. 
