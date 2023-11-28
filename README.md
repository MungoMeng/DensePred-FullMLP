# Full-resolution MLPs for Medical Dense Prediction
Dense prediction is a fundamental requirement for many medical vision tasks such as medical image restoration, registration, and segmentation. The most popular vision model, Convolutional Neural Networks (CNNs), has reached bottlenecks due to the intrinsic locality of convolution operations. Recently, transformers have been widely adopted for dense prediction for their capability to capture long-range visual dependence. However, due to the high computational complexity and large memory consumption of self-attention operations, transformers are usually used at downsampled feature resolutions. Such usage cannot effectively leverage the tissue-level textural information available only at the full image resolution. This textural information is crucial for medical dense prediction as it can differentiate the subtle human anatomy in medical images. **In this study, we hypothesize that Multi-layer Perceptrons (MLPs) are superior alternatives to transformers in medical dense prediction where tissue-level details dominate the performance, as MLPs enable long-range dependence at the full image resolution.** To validate our hypothesis, we develop a full-resolution hierarchical MLP framework that uses MLPs beginning from the full image resolution.  

## Notification
The official code will be released upon paper publication.

## Publication
For more details, please refer to our paper:
* **Mingyuan Meng, Yuxin Xue, Dagan Feng, Lei Bi, and Jinman Kim, "Full-resolution MLPs Empower Medical Dense Prediction," Under Review. [[arXiv]()]**
