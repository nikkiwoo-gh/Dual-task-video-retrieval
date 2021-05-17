# Dual-task-video-retrieval
This is the implementation of the dual-task model presented in Jiaxin Wu, Chong-Wah Ngo, Interpretable Embedding for Ad-Hoc Video Search, ACM Multimedia (ACM MM) 2020.

##Set up
The code is built on python 2.7 and torch 1.4. But it also goes with python 3.6. Other required packages please see requirement.txt.

# training
please refer to do_train_dual_task.sh. 

# test
please refer to do_prediction_iacc.3.sh for the IACC.3 dataset and do_prediction_v3c1.sh for the V3C1 dataset.
The paths needs to be properly set.

# Evalution
please use tv-avs-eval/do_eval_iacc.3.sh and tv-avs-eval/do_eval_v3c1.sh for evaluation.
The paths needs to be properly set.


