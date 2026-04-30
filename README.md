# 🧠 LLM-Based Cancer Medical Chatbot (BioMistralCancer)

## 📌 Overview

This project presents the development of a text question answering (TQA) system for cancer-related medical queries using a Large Language Model (LLM) approach.

The system is built upon the BioMistral language model and fine-tuned using a filtered version of the MedAlpaca dataset focused specifically on cancer-related information. The resulting model, named **BioMistralCancer**, is designed to provide accurate and relevant responses to medical questions in the cancer domain.

This project is developed as part of a Master's thesis in Mathematics (Data Science) and integrates deep learning, natural language processing, and web-based deployment.

---

## 🎯 Objectives

* Develop a cancer-focused medical chatbot using LLM
* Improve model performance through domain-specific fine-tuning
* Evaluate model performance based on accuracy and response quality
* Deploy the model into a real-time web-based application

---

## 🧠 Methodology

### 1. Data Collection

* Dataset: MedAlpaca (filtered for cancer-related data)

### 2. Data Preprocessing

* Handling missing values
* Removing irrelevant characters
* Filtering dataset for cancer-specific content

### 3. Model Development

* Base model: BioMistral (biomedical LLM)
* Fine-tuning on cancer-specific dataset
* Implementation of **Low-Rank Adaptation (LoRA)** for efficient training

### 4. Evaluation

Model performance is evaluated using:

* Accuracy (MedQA, MedMCQA, PubMedQA datasets)
* BLEU Score (CancerGov dataset)
* Runtime training efficiency

### 5. Deployment

* Web-based chatbot using Django framework
* Real-time user interaction through browser interface

---

## 📊 Results

* BioMistralCancer improves accuracy by **>10%** compared to the base BioMistral model 
* Outperforms other biomedical models such as BioGPT and ClinicalGPT 
* LoRA reduces training cost and improves runtime efficiency (≈2x faster) 
* Achieves strong performance across multiple QA datasets (MedQA, MedMCQA, PubMedQA)

---

## 💬 Features

* Cancer-related question answering system
* Real-time chatbot interaction
* Web-based interface
* Domain-specific medical responses

---

## 🛠 Tech Stack

* Python
* Transformers / Hugging Face
* BioMistral (LLM)
* LoRA (PEFT)
* Django (Web Framework)

---

## ⚠️ Disclaimer

This system is developed for research and educational purposes only.
It is not intended to replace professional medical advice or diagnosis.

---

## 🚀 Contribution

This project demonstrates how domain-specific large language models can be applied in healthcare, particularly in improving access to cancer-related medical information and supporting clinical decision-making.

---

## 📁 Project Structure

* `/model` → training & fine-tuning code
* `/dataset` → processed dataset
* `/webapp` → Django chatbot application
* `/evaluation` → testing results & metrics
* `/docs` → thesis report

---

## 👤 Author

Sayyid Abdullah
Master of Mathematics – Data Science
