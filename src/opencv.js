// Record if the model have been loaded.
let cvModels = [];

// Run test only after OpenCV.js loaded.
let onOpenCvReady = function() {
  cv.onRuntimeInitialized = () => {
    updateStatusPicture(runtimeStatus, 1);
    runButton.disabled = false;
  }
}

// The whole predict pipeline.
async function cvPredict(model) {
  let labels = await loadLabels(model.label);
  let net = await cvLoadModel(model);
  let inputMat = getInputMat();
  let input = cv.blobFromImage(inputMat, 1, new cv.Size(224, 224), new cv.Scalar(0, 0, 0));

  console.log('OpenCV.js start inference...');
  net.setInput(input);
  cvExcute(net, labels);
}

// Load onnx model.
async function cvLoadModel(model) {
  let index = cvModels.indexOf(model.path);
  if(index == -1) {
    updateStatusPicture(modelStatus, 0);
    let modelArrayBuffer = await loadModel(model.path);
    let modelUint8Array = new Uint8Array(modelArrayBuffer);
    cv.FS_createDataFile('/', model.name, modelUint8Array, true, false, false);
    updateStatusPicture(modelStatus, 1);
    cvModels.push(model.path);
  } else {
    updateStatusPicture(modelStatus, 1);
  }
  let net = cv.readNetFromONNX(model.name);

  return net;
}

// Excute model one by one.
async function cvExcute(net, labels, iteration = 0) {
  let start = performance.now();
  let result = await cvExcuteSingle(net);
  let end = performance.now();
  let delta = end - start;
  let resultData = result.data32F;
  let classes = getTopClasses(resultData, labels);
  printResult(classes);
  timeSum.push(delta);
  updateIterationProgress(iteration);
  console.log(`Iterations: ${iteration+1} / ${iterations+1}, inference time: ${delta}ms`);

  ++iteration;
  if(iteration < iterations+1) {
    setTimeout(function() {
      cvExcute(net, labels, iteration);
    }, 0);
  } else {
    console.log('Test finished!');
    updateResult(classes);
    net.delete();
    result.delete();
    runButton.disabled = false;
  };
}

// Excute single model.
async function cvExcuteSingle(net) {
  let result = net.forward();
  return result;
}

// Read the image from webpage.
// The images have to be loaded in to a range of [0, 1].
// Normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
function getInputMat() {
    let mat = cv.imread("imgsrc");
    let matC3 = new cv.Mat(mat.matSize[0], mat.matSize[1], cv.CV_8UC3);
    cv.cvtColor(mat, matC3, cv.COLOR_RGBA2RGB);
    let matdata = matC3.data;
    let stddata = [];
    for(var i=0; i<mat.matSize[0]*mat.matSize[1]; ++i) {
      stddata.push((matdata[3*i]/255-0.485)/0.229); 
      stddata.push((matdata[3*i+1]/255-0.456)/0.224);
      stddata.push((matdata[3*i+2]/255-0.406)/0.225);
    };
    let inputMat = cv.matFromArray(mat.matSize[0], mat.matSize[1], cv.CV_32FC3, stddata);

    return inputMat;
}
