# H--8b_CNN
This repository is for creating a convolutional neural network for seperating signal (H->8b) decay from background (Wjets). It contains python programs for creating images, training the model + prediction, and outputing distribution and ROC curves. 

## How it works
### Step 1
The first step is to create a text file with filepaths of input files for image creation. These input files need to be in mini aod format, and need to contain information about the AK15 jet. These files should also include in their name, "wjets" for background, and "8b" for signal. 
### Step 2
Now, use the text file from step 1 as an input to image creation. This can be done in the multi_channel_multi_weight_img_creation.py program by replacing "input_2.txt" with your input text file name in the line: "with open("input_2.txt") as f:" . Then, run the multi_channel_multi_weight_img_creation.py python file. The output will be a root file containing the created images of the 3 channels with their correct label for signal/background, and their weights if applicable. 
### Step 3
We are now ready for the training process. In the CNN_train_weighted_3channel.py file, replace the input file for the root file you created in step 2 (line 15 file_path), and run the python program. The output will be a root file, a model and an png image for the ROC curve. 
### Step 4
If you want to plot the distribution, just replace the input file name (~line 9) in CNN_distribution.py with the output root file from your Step 3, and run the python program. The output will be a png image that displays the distribution of signal/background for the model.
