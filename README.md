<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Useful App Review Classification</h3>
    <br />
  <p align="center">
    This machine learning project utilizes the IndoBERT-BiLSTM model to automate the classification of app reviews into "Useful" or "Not Useful" categories. A "Useful Review" is defined as one that provides constructive feedback and valuable insights to aid developers in improving their applications. Experiment results demonstrate that the IndoBERT-BiLSTM model achieves the best performance with an accuracy rate of 95.49%, representing a 1.16% improvement over the fine-tuned IndoBERT model.
    <br />
    <br />
    <a href="https://huggingface.co/spaces/kadabengaran/useful-review-classification">View Demo</a>
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

Useful App Review Classification aims to address the challenge of evaluating the quality and performance of applications based on user reviews. In today's world, where the volume of app usage and the number of reviews are increasing rapidly, not all reviews are constructive. Some contain only praise or criticism without offering valuable feedback.

<p align="center">
 <a href="#">
    <img src="https://i.ibb.co/6yz7vMt/image.png" alt="What is useful review" height="500">
  </a>
</p>

Manually sorting through thousands of reviews is a time-consuming and labor-intensive task. To overcome this challenge, we employ an efficient and effective method for identifying useful reviews. This research project utilizes the IndoBERT word embedding method and a BiLSTM classifier for review classification..

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MODEL ARCHITECHTURE -->
## Model Architecture

<p align="center">
 <a href="#">
    <img src="https://i.ibb.co/7GD4LQ7/arsitektur-model.png" alt="Model Architecture" height="300">
  </a>
</p>

The approach involves using the IndoBERT-BiLSTM model. The input sentences, tokenized for processing, are fed into the IndoBERT model to obtain output, which is the last hidden state of IndoBERT for each token in sequence. These sequences are then passed through a BiLSTM layer to capture sequential information. The final hidden state from the BiLSTM layer undergoes dropout layers to prevent overfitting and enhance generalization. Finally, the output layer performs a linear transformation to generate logits, the model's output for the classification task.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Technologies Used

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* Pytorch
* Huggingface Transformers
* Streamlit

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Technologies Used

The website application, deployed using Streamlit and Hugging Face Space, offers several powerful features:

* **Single Input Classification**: Easily classify a single text review to determine whether it falls into the category of "Useful Review" or not. This feature simplifies the evaluation of individual reviews.
<p align="center">
 <a href="#">
    <img src="https://i.ibb.co/Xxy9Vn5/web-single.png" alt="Single Input Classification"
  </a>
</p>

* **Multiple Input Classification**: Streamline the review classification process by uploading multiple text reviews in a CSV format. The application will efficiently classify these reviews and present the results in a user-friendly table format for easy analysis and interpretation.
<p align="center">
 <a href="#">
    <img src="https://i.ibb.co/FV8r5Y4/web-multi.png" alt="Multiple Input Classification">
  </a>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
