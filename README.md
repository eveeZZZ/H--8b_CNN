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

## Running Tips
Typically, the algorithm runs better on a large data set, and while the training doesn't take the longest (compare to particle net algorithms), the data preparation time could be very long. Here are some tips in general on secure and time-efficient running. 
### Use GPUS
In the Wisconsin computing cluster, ssh into cmsgpu01, which will allow for the use of GPU cores. This can be helpful with training the Neural Networks. 
### Use Screens
Since these algorithms can take hours or even days to run, it is useful to run these algorithms on a screen that will not be affected by anything on the local computer. To do this, first ssh into your hep cluster account, and then instead of running everything directly, you can first do `screen`, which will take you to a different screen. Then, navigate to the correct directory and run your program there. You can do Ctrl+A and then Ctrl+D to exit the screen (while it is running). Then when you want to come back to the screen, in your terminal you can do `screen -ls` to see the available screens you have open, then find the one that you want to go back to and do `screen -r {your screen number}` to go into that screen again. 
