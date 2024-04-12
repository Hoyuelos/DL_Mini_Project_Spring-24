# DL_Mini_Project_Spring-24
Team - Daredevils


<h1> CIFAR-10 classification using Custom Resnet </h1>

Accuracy: 95.16%
Parameters: 4,964,266


<h3> In the repository </h3>
<li> torch.out log file - contains the complete execution outputs of the program </li>
<li> data directory that contains data files </li>
<li> checkpoint directory that contains model's final checkpoint - the one used to generate best result on kaggle </li>
<li> test_nolabels_csv directory that contains the prediction csv file </li>


<h3> Steps to run the code </h3>
<ol>
<li> Make sure to have folder structure same as this repository   </li>
</li> Make sure to keep the pkl file for test set and Cifar dataset folder (unziped) in the data directory (otherwise just change the path for both the datasets in the begening of the code) </li>
<li> Run requirements.txt to install all the required dependencies </li>
<li> Run resNet.py python file</li>
<li> Predictions will be saved in test_nolabels_csv folder. This csv is to be uploaded on Kaggle to get test accuracy</li>
</ol>
