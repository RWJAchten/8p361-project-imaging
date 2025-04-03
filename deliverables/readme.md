# Final project code for 8P361 (2024-3) Project AI for medical image analysis
Code used to train machine learning models and gather results used in the final report.

## contributors
Robin Achten, 1874551 \
Nienke Sengers, \
Nika Vredenbregt, \
Julien Vermeer, \

Group 7

## installation
1. Clone or download the respository
2. Include the 'Data' directory in the right location (this is NOT included in the deliverables).
3. If you are planning on running or editing the code, it is advised you satisfy the _requirements_ (see next section)

## requirements
The used library versions and dependencies can be found in **requirements.txt**. Note that especially older versions of tensorflow==2.10 and numpy==1.25.1 have been used as only these older versions are compatible with GPU acceleration using CUDA. 

## general respository outline
As mentioned, the data containing folder named 'Data' should be placed in the respository such that it matches the architecture:

parent (this respository)\
|\
|--data--|--train+val--|--train\
|&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp;&nbsp;|-- valid\
|--hybrid\
|--results\
|--result_logs

the other folders include:

parent (this respository):
- hybrid \
&emsp;  - **Controller.tsv** (contains all functions necessary for training and evaluating the models.) \
&emsp;  - **super_scheduler.ipynb** (the notebook where all results have been computed) \
&emsp;  - **Model_builds.py** (contains all model architectures) \
&emsp;  - **MBConv.ipynb** (contains the inverted residual block)
&emsp;  - **ViT.ipynb** (contains the patch embedding, transformer block and MLP) \
- results (the model weights and .json files have been saved here for the final models that were evaluated on the test set.)
- result_logs (contains the tensorboard logs for the final models)

