# Bayesian-Networks
Some work I did on Bayesian Networks for my AI final

A few notes on how to use the files associated with this project:

+ I suggest you read the paper associated with this project before diving into the code (AI_Final.pdf)
    + The context provided by the paper is esspecially helpful we looking through the Bayes Net files
+ You'll need to install the pgmpy package to run the Bayes Net files (v 0.1.7)
+ You may also need to install wrapt v 1.11.2
+ All models were tested on the famous Wisconsin breast cancer dataset
    + I chose this dataset based on the application of simple Bayes nets for medical diagnosis
+ The baseline_model.py compares a couple of simple models
+ The loader.py file is a helper file used to speed the loading of data
+ The visualize.py file runs some basic visualizations of the dataset
    + These were used to get a grip on the distributions followed by each of features
+ The bayes_net.py file creates a series of Bayesian networks using various techniques
    + It's basically a combination of what was previously a series of Bayes files to aid readability
    + NOTE: this file takes a LONG time (up to 5 minutes) to run (not a ridiculous amount of time, but still quite a bit)
    + Runtime is simply not the strength of Bayesian Methods
    + Running this file will output a number of warning messages, these warning messages are caused by the interaction between pgmpy and pandas and are not an issue
+ The make_paper_pgms.py file was used to make graphics for the paper
    + The daft package is required to run it
+ The naive_bayes_pgm.py file runs a naive Bayes model built using pgmpy