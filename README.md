# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
**This dataset contains data about clients from a banking institution obtained via direct marketing campaigns.
The goal of this classification problem is to predict if the client will subscribe to a term deposit.**

**The best performing model was a VotingEnsemble machine learning model obtained using AutoML.**


![Optimizing an Pipeline.png](Optimizing%20an%20Pipeline.png)

## Scikit-learn Pipeline
**The development of the pipeline begins with the creation of a python training script file. Within this file, the raw banking
data is converted from a csv format to a Tabulardataset format. This data set is then passed to cleaning function which cleans
the data and encodes various columns. The cleaned data set is then split into training and testing sets. A LogisticRegression
algorithm uses these data sets along with two user defined parameters, C and max_iter to develop the ML model. This script is
used along with other parameters and policies to create a Hyperdrive configuration. This configuration is written in python code
using the Azure SDK within an Azure notebook. The configuration is then used to run an experiment in Hyperdrive. The best run
which maximized the accuracy primary metric is then saved to a file.**

**The parameter sampler method chosen is the RandomParameterSampling method. The advantage of this method is that it ensures
that results obtained from the sample should approximate what would have been obtained if the entire population had been measured**

**In order to save on compute resources and cost, a Bandit early stopping policy was chosen. The policy early terminates any runs where
the primary metric is not within the specified slack factor amount with respect to the best performing training run. An evaluation interval
and delay evaluation argument was also specified. These affect the frequency for applying the metric and the number of intervals to
delay the first evaluation, respectively**

## AutoML
**When using Azure AutoML to obtain the best model given the provided dataset and parameters, it determined that the best model
was the VotingEnsemble model. This model combines the predictions from multiple other models and assigns a weight to each 
of those models. In this scenario, the model with the highest weight was an XGBoostClassifier model which used SparseNormalizer
for data transformation. This model has quite a bit of hyperparameters like eta, gamma, max_depth_max_leaves and n-estimators
which must be tuned.**

## Pipeline comparison
**The performance of the best hyperdrive and the best AutoML models was very similar. The hyperdrive model had an accuracy
of .9099 while the AutoML model had an accuracy of .9177. The architectures of the pipelines are different in that the 
hyperdrive method requires the user to pick an algorithm and parameters to tune while the AutoML method does not. AutoML
runs various experiments using different algorithms and parameters. A user, however can decide to remove some algorithms
from consideration.**

## Future work
**In order to try and improve the hyperdrive results, additional model parameters can be included in the tuning process.
This may help in that not only the default values are used. Also a different parameter sampler can be tried. Lastly, one can
change the early termination policy to make sure that runs are not being terminated prematurely. In the AutoML pipeline, one can
try and increase the experiment_timeout parameter. This will allow the model more time to find better algorithms and parameters.
Also, you can increase the iteration timeout parameter if you see that some iterations were not completed. Lastly, one can exclude
some models that are noticed to not provide good results.  By doing this, more time is spent on finding and tuning better models.
In the end, one must consider and understand the compute costs incurred by these changes and decide if its worth it.**
