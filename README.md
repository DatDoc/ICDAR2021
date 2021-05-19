# ICDAR2021

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
  
    -   In SIW 2021 ICDAR Competition on Script Identification in the Wild, after
      observing the dataset, the participants assumed that the score of the third task
      is base on the first two, so they decided to turn their attention for maximizing the
      accuracy score of task 1 and task 2 separately. They divided the original problem
      into 2 main phases. First, they built a two-class classifier to classify handwritten
      and printed samples. Then, for each one of them were independently train two
      more classifiers to identify thirteen classes.
      #- Handwritten/Printed type classification: 
      They used EfficientNet-B7 architecture which is pre-trained on ImageNet. Which was stacked 1 fully con-
      nected layer with 1024 units in front of the output layer. A sigmoid loss function
      is used since it is a binary classification. An Adam optimizer with learning rate of
      0.0001 is used for the training process. The dataset will be cut into 2 parts with a
      train/validation rate is 80/20. First, they allowed FC head warm up by freezing
      all layers in the body of the network and train in 20 epochs with a batch size of
      32. After the FC head has started to learn patterns in their dataset, pause train-
      ing, unfreeze from layer 20th of the body, and then continue the training with
      Stochastic Gradient Descent (SGD) and a learning rate of 0.001 in 60 epochs
      with batch size of 16.
Script identification: To create a solid and stable validation for the training
process, they split stratified five folds with each fold contains the same train
validation ratio as the first task (4:1), this process will make sure that each set
contains approximately the same percentage of samples of each target class as
the complete set.

# ICDAR2021 Competition (end)
