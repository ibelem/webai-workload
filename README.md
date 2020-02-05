# Web AI Workload

Web AI Workload will be run against the most prominent open source ML Javascript library in the same market.

## How to run
1. Download model file `squeezenet1.1.onnx` `MobileNet v2-1.0.onnx` `resnet50v1.onnx` `resnet50v2.onnx` from [ONNX Model Zoo](https://github.com/onnx/models). 
  
    Rename and put them in the model folder.
    ```
    mv squeezenet1.1.onnx squeezenet.onnx
    mv MobileNet v2-1.0.onnx mobilenetv2.onnx
    ```

2. Start an http server in this folder. You can install [`http-server`](https://github.com/indexzero/http-server) via:
    ```
    npm install http-server -g
    ```
    Then start an http server by running
    ```
    http-server -c-1 -p 3000
    ```

    This will start the local http server with disabled cache and listens on port 3000

3. Open up the Chrome and access this URL:

    chrome://flags

    Search 'SIMD' and enable WebAssembly SIMD support. Then relunch the Chrome.

4. Open up the Chrome and access this URL:
    
    http://localhost:3000/

5. Click on Run button to see results of the test.
