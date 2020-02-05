// The inference session with ONNX.js.
let session = {};

// Run test only after ONNX.js loaded.
let onOnnxjsReady = function() {
  let backend = window.sessionStorage.getItem("backend") || backendName.value;

  // onnx.backend.wasm.cpuFallback = false;
  if(backend == 'WASM') {
    onnx.backend.wasm.worker = 0;
  } else if (backend == 'Threads') {
    onnx.backend.wasm.worker = window.navigator.hardwareConcurrency - 1;
  }
  
  updateStatusPicture(runtimeStatus, 1);
  runButton.disabled = false;
}

// The whole predict pipeline.
async function onnxPredict(model) {
  runButton.disabled = true;

  session = new onnx.InferenceSession({backendHint: 'wasm'});

  let labels = await loadLabels(model.label);
  await session.loadModel(model.path);
  updateStatusPicture(modelStatus, 1);

  let inputTensor = await getInputTensor(imgElement);

  console.log('OpenCV.js start inference...');
  onnxExcute(inputTensor, labels);
}

// Excute model one by one.
async function onnxExcute(input, labels, iteration = 0) {
  let start = performance.now();
  let result = await onnxExcuteSingle(input);
  let end = performance.now();
  let delta = end - start;
  let resultData = result.values().next().value.data;
  let classes = getTopClasses(resultData, labels);
  printResult(classes);
  timeSum.push(delta);
  updateIterationProgress(iteration);
  console.log(`Iterations: ${iteration+1} / ${iterations+1}, inference time: ${delta}ms`);

  ++iteration;
  if(iteration < iterations+1) {
    setTimeout(function() {
      onnxExcute(input, labels, iteration);
    }, 0);
  } else {
    console.log('Test finished!');
    updateResult(classes);
    result.delete();
    runButton.disabled = false;
  };
}

// Excute single model.
async function onnxExcuteSingle(input) {
  let result = await session.run([input]);
  return result;
}

// The images have to be loaded in to a range of [0, 1].
// Normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
async function getInputTensor(imageElement) {
  let width = 224;
  let height = 224;
  let imageData = getImageData(imageElement, width, height);

  const dataFromImage = ndarray(new Float32Array(imageData), [width, height, 4]);
  const dataProcessed = ndarray(new Float32Array(width * height * 3), [1, 3, height, width]);

  // Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
  ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 0));
  ndarray.ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
  ndarray.ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 2));

  // Normalize 0-255 to 0-1.
  ndarray.ops.divseq(dataProcessed, 255.0);

  // Normalized using mean = [0.485, 0.456, 0.406].
  ndarray.ops.subseq(dataProcessed.pick(0, 0, null, null), 0.485);
  ndarray.ops.subseq(dataProcessed.pick(0, 1, null, null), 0.456);
  ndarray.ops.subseq(dataProcessed.pick(0, 2, null, null), 0.406);

  // Normalized using std = [0.229, 0.224, 0.225].
  ndarray.ops.divseq(dataProcessed.pick(0, 0, null, null), 0.229);
  ndarray.ops.divseq(dataProcessed.pick(0, 1, null, null), 0.224);
  ndarray.ops.divseq(dataProcessed.pick(0, 2, null, null), 0.225);

  let inputTensor = new onnx.Tensor(dataProcessed.data, 'float32', [1, 3, 224, 224]);

  return inputTensor;
}
