# convCL
convCL is a set of OpenCL-based convolution algorithms, such as GEMM, Gemm Implicit or Winograd. This methods supports multi-GPU execution, using channel (convCL_oneimg.py) or object (convCL.py) level parallelism. 
An example.py can be executed as follows:

**python3 examples.py 0 1 256 256 256 512 1 5 5 GEMM**

which will execute GEMM method with the one image of the size *256 X 256 X 256* as input data and  *512* filters using *1* GPU on the *0* platform. 
The result of the script shows a relative deviation of the result tensor sum compared to the PyTorch result tensor sum.
 

