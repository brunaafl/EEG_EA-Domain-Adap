# EEG_EA-Domain-Adap
Paper's repository

This is a repository containing code for the paper titled "A Systematic Evaluation of Euclidean Alignment with Deep Learning for EEG Decoding". The paper explores the use of Euclidean Alignment (EA) in Deep Learning and how it impacts the Electroencephalogram (EEG) decoding process and transfer learning.

## Getting Started

To run the code, you will need to create a virtual environment with the following packages:

```bash
conda create -n eeg python=3.9
conda activate eu_al_eeg
pip install -r requirements.txt
```
## Running the Code
Once you have set up the virtual environment, you can run the code using the following command:

```bash
python src/Bruna/script_moabb_exp1.py --config_file config/config.yaml --alignment 'alignment' --num_exp 'exp_1' --dataset 'BNCI2014001'
```
Here is a brief explanation of the command line arguments:

* alignment: specifies the alignment method to use. It can be "alignment", "no-alignment", or "r-alignment"
* num_exp: specifies the number of the experiment. It can be "exp_1" or "exp_4"
* dataset: specifies the dataset to use. It can be 'BNCI2014001' or 'PhysionetMI'.
* config_file: specifies the path to the yaml config file.
* session: could be "both", "session_T", or "session_E".
  
