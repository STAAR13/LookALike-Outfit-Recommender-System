# 👗 Lookalike Outfit Fashion Recommender System

A **personalized, image-based fashion recommender** system that helps users find visually similar outfits by uploading an image, using **deep learning and nearest neighbour search**.

---

## 📖 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Conclusion](#conclusion)


---

## 🪡 Introduction

With the rise of e-commerce, customers often feel overwhelmed by the vast range of outfit choices. Unlike traditional recommender systems that rely on user purchase history, **this project enables customers to upload any outfit image they like and receive visually similar outfit recommendations**.

This system helps:
- Users find lookalike outfits instantly.
- E-commerce platforms increase product discoverability.
- Fashion enthusiasts explore products visually.

---

## ✨ Features

✅ Upload any outfit image and get top 5 visually similar outfit recommendations.  
✅ Uses **ResNet50 with fine-tuning** for feature extraction.  
✅ **Cosine Similarity with Nearest Neighbours** for fast, relevant retrieval.  
✅ Simple, clean **Streamlit web interface**.  
✅ Transfer learning to handle small fashion datasets effectively.

---

## 🧩 Methodology

1. **Feature Extraction**  
   - Uses **ResNet50** (fine-tuned) to extract feature embeddings from images.

2. **Inventory Embedding**  
   - Passes the Kaggle Fashion Product Images Dataset through the network to generate embeddings.

3. **Recommendation Generation**  
   - Uses **Scikit-learn’s Nearest Neighbours** with **Cosine Similarity** to find the most visually similar outfits to the uploaded image.

---

## 📂 Dataset

- [Kaggle Fashion Product Images Dataset (Small - 572 MB)](https://www.kaggle.com/datasets)
- [Kaggle Fashion Product Images Dataset (Large - 15 GB)](https://www.kaggle.com/datasets)
- **DeepFashion dataset (44,441 images)** used for pre-training to overcome small dataset limitations.
![Dataset Preview](/project_snippets/dataset.png)

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/STAAR13/LookALike-Outfit-Recommender-System.git
cd lookalike-fashion-recommender
```

Install the required packages:

```bash
pip install -r requirements.txt
```
## Usage

To run the web application:

```bash
streamlit run main.py
```
After running the command, the application will start a local web server and automatically open in your default browser.

**How to use:**

1. Click on "Upload Image" to select a fashion item image
2. The system will process the image and extract features
3. View the top 5 recommended similar items from the inventory
4. Explore the recommendations and their details



## Results

Our model achieves high accuracy in finding visually similar fashion items. Some example recommendations:

![Output screen](/project_snippets/result.png)


The system successfully captures:
- Color patterns
- Textures
- General style
- Item category

## Built With

- [OpenCV](https://opencv.org/) - Computer vision library
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [scikit-learn](https://scikit-learn.org/) - Machine learning library
- [Streamlit](https://streamlit.io/) - Web application framework
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [Pillow](https://python-pillow.org/) - Image processing

## Conclusion

This fashion recommender system represents a significant advancement in personalized shopping experiences by leveraging cutting-edge computer vision and machine learning techniques. Our image-based approach eliminates the dependency on purchase history or user ratings, instead focusing on visual similarity to deliver intuitive recommendations that match users' aesthetic preferences.

Key achievements of this project include:
- Successful implementation of a ResNet50-based feature extraction pipeline
- Development of an efficient nearest-neighbor recommendation engine
- Creation of a user-friendly interface for seamless interaction
- Demonstration of high accuracy in matching visual styles and patterns

The system opens several exciting future directions:
- Integration with augmented reality for virtual try-ons
- Expansion to include multi-item outfit recommendations
- Incorporation of seasonal trends and personal style preferences
- Potential for mobile deployment to assist in-store shopping

By bridging the gap between visual inspiration and product discovery, this technology has the potential to transform how consumers interact with fashion e-commerce platforms, making personalized styling accessible to everyone regardless of their fashion expertise.
