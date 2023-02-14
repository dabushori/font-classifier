# Font Classifier
This is the `font-classifier` project for the computer vision course. To read more about the project, read the report located [here](./report/Font%20Classification%20Project.pdf).

# How to run the model and generate the test results?
1. Install all the dependencies:
    * numpy
    * matplotlib
    * h5py
    * pytorch (Pay attention to install the CUDA version. The code will also work without it, but won't finish in reasonable time)
    * open-cv
2. Download the pre-trained model from [this link](https://drive.google.com/file/d/1IiJfCTw0aWa0G8zx9LSbi0d5P7jhr6FJ/view?usp=sharing), unzip it, and replace the empty file in `models/all_models_without_perms` with it.
3. Download the test data from [here](https://drive.google.com/drive/folders/1hmPI7KaWcv-OLwJEQvMNjbOu9IhU_7CR) and replace the empty file in `Project - Test Set\SynthText_test.h5` with it.
4. Run the `model_testing.py` file.

# File structure
* The `Project` and `Project - Test Set` folders contains a template for the train and test datasets. There are empty files that should be replaced with the real train and test datasets.
* The `models` folder is a templaye for the pre-trained model. There's an empty file that should be replaced with the real trained model weights. Any other model that will be trained will be located in this folder.
* The `outputs` folder contains the output of the `model_training.py` script when the pre-trained model was trained. Any other model that will be trained will be located in this folder.
* The `report` folder is the most important one - it contains the report and the test labels in the required csv file format.