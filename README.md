# Generalized Image Classification

Code to support my general process for creating an image classification models

- logistic inception-v3 model
    clone repo to project directory
    separate labeled images into folders
    place images in 'data/images' directory
    run model_builder notebook


TODO:

add support for .png and .bmp

add ability to pass different classifiers to ModelMaker
build in ability to save the model
add option to separate validation set

for generalizeability:
change method names to transform() ->[preprocessing and dataframe creation], fit() ->[test-train split crossval and fit ML model], predict()/evaluate()
