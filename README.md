# SIW ICDAR2021

## Team UIT in SIW ICDAR2021

## Information of two co-authors: 
  ### 1. First author: 
          Name: Nhat Pham Le Quang
          Email: 18520120@gm.uit.edu.vn
          Organization: University of Information Technology VNU-HCM, VietNam
  ### 2. Second author:
          Name: Dat Nguyen Phuc
          Email: 18520573@gm.uit.edu.vn
          Organization: University of Information Technology VNU-HCM, VietNam
          
## Our method in competion:
  
        In SIW 2021 ICDAR Competition on Script Identification in the Wild, after observing the dataset, the participants
      assumed that the score of the third task is base on the first two, so they decided to turn their attention for 
      maximizing the accuracy score of task 1 and task 2 separately. They divided the original problem into 2 main phases. 
      First, they built a two-class classifier to classify handwritten and printed samples. Then, for each one of them were         
      independently train two more classifiers to identify thirteen classes.
        
      **Handwritten/Printed type classification:** 
      
        They used EfficientNet-B7 architecture which is pre-trained on ImageNet. Which was stacked 1 fully connected layer 
      with 1024 units in front of the output layer. A sigmoid loss function is used since it is a binary classification. An 
      Adam optimizer with learning rate of 0.0001 is used for the training process. The dataset will be cut into 2 parts with 
      a train/validation rate is 80/20. First, they allowed FC head warm up by freezing all layers in the body of the network 
      and train in 20 epochs with a batch size of 32. After the FC head has started to learn patterns in their dataset, pause 
      training, unfreeze from layer 20th of the body, and then continue the training with Stochastic Gradient Descent (SGD) 
      and a learning rate of 0.001 in 60 epoch with batch size of 16.
        
      **Script identification:**
      
        To create a solid and stable validation for the training process, they split stratified five folds with each fold 
      contains the same train validation ratio as the first task (4:1), this process will make sure that each set contains 
      approximately the same percentage of samples of each target class as the complete set.
        In order to efficiently distinguish a large amount of imbalanced classes, we extracted the embedding feature from the 
      pooling layer of various CNN backbone models. To address different image size, they trained their models on fixed image 
      size with horizontal flip, vertical flip, then normalize the images by the mean and standard deviation of the ImageNet 
      dataset before feeding them into a pre-trained backbone. Our highest score on the leaderboard is an ensemble of three 
      models using the following: ViT, EfficientNet-B4, ResNeXt50.
        Each model in phase two is trained for 10 epochs with a cosine annealing scheduler having 10 warm-up iterations. We 
      use AdamW optimizer with learning rate of 0.0001 and weight decay of 0.000001 across all models. We optimize using the 
      Crossentropy loss.
      
      **Post-processing steps:**
      
        The first technique is test-time augmentation (TTA). Instead of letting the model predict a single image, we augmented
      the image two times using the training augmentation pipeline (see in paragraph Script identification). So the model has 
      to predict the same image three times but in various forms. This method helps the model give a solid result.
        The second technique is pseudo-labeling. It is used during the test-set inference step. Using the confidence score of 
      the final predictions, if the score of a prediction is greater or equal to the threshold - which is 0.99 for highest 
      score on leaderboard, we will add that image to the train set and start the training process from all over again. 
      Pseudo-labeling not only increase the size of our dataset, but also boost the result of our team significantly.

## Result:

      1. Task 1: 95.25% (f1-score)
      2. Task 2: 92.77% (f1-score)
      3. Task 3: 91.34% (f1-score)

# ICDAR2021 Competition (end)
