# Selu activation
## How to run
* Generate onnx file for the network with `train.py --export-onnx --epochs 1`
* Compile project from the root: `mkdir builld && cd build && cmake .. && make -j15`
* Run with `LD_LIBRARY_PATH="." ./sample_onnx_mnis_tselu --datadir /path/to/mnist/data`

## Expected output
[04/15/2021-05:59:39] [I] Input:
[04/15/2021-05:59:39] [I]
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@++-  =*@@@@@@@@@@@
@@@@@@@@@-      .%@@@@@@@@@@
@@@@@@@@-   .=-  +@@@@@@@@@@
@@@@@@@#  .%@@@. =@@@@@@@@@@
@@@@@@@* .%@@@+  #@@@@@@@@@@
@@@@@@@@%@@@@@.  #@@@@@@@@@@
@@@@@@@@@@@@@+  :@@@@@@@@@@@
@@@@@@@@@@@@*   #@@@@@@@@@@@
@@@@@@@@@@@@=  =@@@@@@@@@@@@
@@@@@@@@@@@-  -@@@@@@@@@@@@@
@@@@@@@@@@%   %@@@@@@@@@@@@@
@@@@@@@@@@:  =@@@@@@@@@@@@@@
@@@@@@@@@*  :@@@@@@@@@@@@@@@
@@@@@@@@@:  =@@@@@@@@@@@@@@@
@@@@@@@@=  -@@@@@@@@@@@@@@@@
@@@@@@@@   %@@@@@@@@@@@@@@@@
@@@@@@@@   %@@@@@@@@@%====@@
@@@@@@@@        -=-       +@
@@@@@@@@-              -++#@
@@@@@@@@@++++-   -++%@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@

[04/15/2021-05:59:39] [I] Output:
[04/15/2021-05:59:39] [I]  Prob 0  0.0000 Class 0:
[04/15/2021-05:59:39] [I]  Prob 1  0.0000 Class 1:
[04/15/2021-05:59:39] [I]  Prob 2  1.0000 Class 2: **********
[04/15/2021-05:59:39] [I]  Prob 3  0.0000 Class 3:
[04/15/2021-05:59:39] [I]  Prob 4  0.0000 Class 4:
[04/15/2021-05:59:39] [I]  Prob 5  0.0000 Class 5:
[04/15/2021-05:59:39] [I]  Prob 6  0.0000 Class 6:
[04/15/2021-05:59:39] [I]  Prob 7  0.0000 Class 7:
[04/15/2021-05:59:39] [I]  Prob 8  0.0000 Class 8:
[04/15/2021-05:59:39] [I]  Prob 9  0.0000 Class 9:
[04/15/2021-05:59:39] [I]
&&&& PASSED TensorRT.sample_onnx_mnist_selu # ./sample_onnx_mnis_tselu --datadir ../samples/opensource/sampleOnnxMNISTselu/