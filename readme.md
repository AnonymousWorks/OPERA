## OPERA: Don't Miss the Model Loading in Deep Learning Compiler Testing
OPERA is a test-migration-based technique to improve the testing of the model loading stage of DL compilers.
It considers different sources of test inputs in DL libraries for migration.
Also, it designs a diversity-based test prioritization strategy to migrate and execute those test inputs that are more likely to detect diverse bugs in the model loading stage, to improve the testing efficiency.


### Reproducibility

####  1. Build Environment
> Install the test object and migration objects as follows:
* TVM v0.13:
     Please refer to [the official documentation](https://tvm.apache.org/docs/install/from_source.html) to install it from the source. You can execute `git checkout b48fcab` to get the same version as this work before installing it.
* PyTorch v1.7:
     `pip install torch==1.7.0`
* Keras v2.3:
    `pip install keras==2.3.1`
* ONNX 1.8:
    `pip install onnx==1.8.0`


#### 2. Extract Operator Instances
> instruments each operator API in the source code of DL compilers for operator instance extraction.
###### Steps:
  1) execute all test suites equipped by the three DL libraries and the DocTer fuzzing under the instrumented DL libraries.
  2) save the operator instance and wrap them into DL models automatically in the designed instrumented code,
  3) The extracted test inputs were saved in the path library_name/data (e.g., keras/data)


####  3. Run Test Prioritization
Execute the prioritization to rank all migrated test inputs.
```
python run_tcp.py
```
The result was saved in the current path (e.g., ranked_test_case_keras.py)


----

### Supplement Results(RAUC-k value)
$RAUC-k$ is a metric to measure the prioritization effectiveness when all prioritized tests can not be executed completely in a limited time practically. 
Therefore, RAUC-s are proposed to measure the prioritization effectiveness when only top k tests can be executed.
Specifically, it is calculated based on the prioritization result graph with the number of tests as the x-axis and the bug number as the y-axis.
The RAUC is determined by calculating the area under the curve of the prioritization technique and contrasting it with the area under the curve of the ideal prioritization, which represents the sequential order in which the test cases would have been executed had all bugs been known beforehand.
In our study, we evaluated the performance of the TCP technique on different proportions of test cases, specifically 25\%, 50\%, 75\%, and 100\% of the total number of tests, which we referred to as RAUC-25\%, RAUC-50\%, RAUC-75\%, and RAUC-100\% respectively. A higher value of RAUC-k indicates better performance of the prioritization strategy. The bold in the table means the best value.
![rauc](https://github.com/AnonymousWorks/OPERA/assets/89679728/d54d1f61-1b07-4e6e-9dbb-7a0290e8297b)



### Bug Details
> This work has detected 159 previously unknown bugs, 80 of which have been confirmed/fixed by developers.
The id of each bug detected by OPERA will be released after the paper review due to the double-blind policy.


