# Emotion-recognition

---
Facial expression emotion recognition is an intuitive reflection of a person’s mental state, which contains rich emotional information, and is one of the most important forms of interpersonal communication. It can be used in various fields, including psychology. As a celebrity in ancient China, Zeng Guofan’s wisdom involves facial emotion recognition techniques. His book Bing Jian summarizes eight methods on how to identify people, especially how to choose the right one, which means “look at the eyes and nose for evil and righteousness, the lips for truth and falsehood; the temperament for success and fame, the spirit for wealth and fortune; the fingers and claws for ideas, the hamstrings for setback; if you want to know his consecution, you can focus on what he has said.” It is said that a person’s personality, mind, goodness, and badness can be showed by his face. However, due to the complexity and variability of human facial expression emotion features, traditional facial expression emotion recognition technology has the disadvantages of insufficient feature extraction and susceptibility to external environmental influences. Therefore, this article proposes a novel feature fusion dual-channel expression recognition algorithm based on machine learning theory and philosophical thinking. Specifically, the feature extracted using convolutional neural network (CNN) ignores the problem of subtle changes in facial expressions. The first path of the proposed algorithm takes the Gabor feature of the ROI area as input. In order to make full use of the detailed features of the active facial expression emotion area, first segment the active facial expression emotion area from the original face image, and use the Gabor transform to extract the emotion features of the area. Focus on the detailed description of the local area. The second path proposes an efficient channel attention network based on depth separable convolution to improve linear bottleneck structure, reduce network complexity, and prevent overfitting by designing an efficient attention module that combines the depth of the feature map with spatial information. It focuses more on extracting important features, improves emotion recognition accuracy, and outperforms the competition on the FER2013 dataset.

---

## Experimental Data Set

The data set used in this article is FER-2013. Due to the small amount of data in the original facial expression data set, it is far from enough for data-driven deep learning, so data augmentation is a very important operation. In the network training stage, in order to prevent the network from overfitting, I first do a series of random transformations, including flipping, rotating, cutting, etc., and then transform the data image size to 48 × 48 size, finally the output classification with the highest score is the corresponding expression. This method can expand the size of the data set, make the trained network model more generalized and robust, and further improve the accuracy of recognition.


---

## Evaluation Method
The overall accuracy rate is used as the evaluation index of this study, and its calculation formula is as follows:

# Acc=TP+TN/(TP+TN+FP+FN)

where TP represents the positive samples predicted by the model as positive, TN represents the negative samples predicted by the model as negative, FP represents the negative samples predicted by the model as positive, and FN represents the positive samples predicted by the model as negative.
