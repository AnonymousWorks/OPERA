## OPERA: Don't Miss the Model Loading in Deep Learning Compiler Testing
OPERA is a test-migration-based technique to improve the testing of the model loading stage of DL compilers.
It considers different sources of test inputs in DL libraries for migration (i.e., test inputs documented in DL libraries and test inputs generated using recent fuzzers).
Also, it designs a diversity-based test prioritization strategy to migrate and execute those test inputs that are more likely to detect diverse bugs in the model loading stage, to improve the testing efficiency.


### Reproducibility

####  1. Build Environment

#### 2. Extract Operator Instances

####  3. Run Test Prioritization



### Supplement Results(RAUC-k value)
$RAUC-k$ is a metric to measure the prioritization effectiveness when all prioritized tests can not be executed completely in a limited time practically. 
Therefore, RAUC-s are proposed to measure the prioritization effectiveness when only top k tests can be executed.
Specifically, it is calculated based on the prioritization result graph with the number of tests as the x-axis and the bug number as the y-axis.
The RAUC is determined by calculating the area under the curve of the prioritization technique and contrasting it with the area under the curve of the ideal prioritization, which represents the sequential order in which the test cases would have been executed had all bugs been known beforehand.
In our study, we evaluated the performance of the TCP technique on different proportions of test cases, specifically 25\%, 50\%, 75\%, and 100\% of the total number of tests, which we referred to as RAUC-25\%, RAUC-50\%, RAUC-75\%, and RAUC-100\% respectively. A higher value of RAUC-k indicates better performance of the prioritization strategy. The bold in the table means the best value.
![image](https://github.com/AnonymousWorks/OPERA/assets/89679728/57c206d4-2c8e-46b6-bbc3-269b18f2a299)


