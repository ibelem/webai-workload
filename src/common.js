const modelZoo = [{
  name: 'squeezenet',
  path: 'https://webaimodel.s3-us-west-2.amazonaws.com/onnx/squeezenet.onnx',
  label: './model/labels1000.txt'
}, {
  name: 'mobilenetv2',
  path: 'https://webaimodel.s3-us-west-2.amazonaws.com/onnx/mobilenetv2.onnx',
  label: './model/labels1000.txt'
}, {
  name: 'resnet50v1',
  path: 'https://webaimodel.s3-us-west-2.amazonaws.com/onnx/resnet50v1.onnx',
  label: './model/labels1000.txt'
}, {
  name: 'resnet50v2',
  path: 'https://webaimodel.s3-us-west-2.amazonaws.com/onnx/resnet50v2.onnx',
  label: './model/labels1000.txt'
}];

/*************************   UI.js   **************************/

// Update tag value.
function updatInnerHTML(Element, value) {
  Element.innerHTML = value;
}

// Update status picture.
function updateStatusPicture(imgElement, status) {
  if(status == 1) {
    imgElement.src = './icon/loaded.png';
  } else {
    imgElement.src = './icon/loading.png';
  }
}

// Update select box value.
function updateSelectBox(selectElement, value) {
  let options = selectElement.options;

  for(let i=0; i<=options.length; i++) {
    if(options[i].value == value) {
      options[i].defaultSelected = true;
      options[i].selected = true;
      break;
    }
  }
}

// Hide/Show select option;
function setSelectOption(selectElement, value, status) {
  let options = selectElement.options;

  for(let i=0; i<options.length; i++) {
    if(options[i].value == value) {
      if(status == 'hide') {
        options[i].disabled = true;
      } else if(status == 'show') {
        options[i].disabled = false;
      }
      break;
    }
  }
}

// Update the iteration UI during the perdict process.
function updateIterationProgress(calIteration) {
  iterationProgress.style.visibility = 'visible';
  iterationProgress.value = (calIteration)*100/(iterations);
  showIterations.innerHTML = `Iterations: ${calIteration} / ${iterations}`;
}

// Show the result UI after inferance.
function updateResult(classes) {
  let finalResult = summarize(timeSum);
  testInfo.style.visibility="visible";
  testInfo.innerHTML = `<b>ML Runtime:</b> ${libraryName.value} <br>
                        <b>WASM Build:</b> ${backendName.value} <br>
                        <b>Model Path:</b> ${document.getElementById("modelName").value} <br>
                        <b>Inference Time:</b> ${finalResult.mean.toFixed(2)}`;
  if(iterations != 1) {
    testInfo.innerHTML += ` ± ${finalResult.std.toFixed(2)} [ms] <br> <br>`;
  } else {
    testInfo.innerHTML += ` [ms] <br> <br>`;
  };
  testInfo.innerHTML += `<b>label1</b>: ${classes[0].label}, probability: ${classes[0].prob}% <br>
                         <b>label2</b>: ${classes[1].label}, probability: ${classes[1].prob}% <br>
                         <b>label3</b>: ${classes[2].label}, probability: ${classes[2].prob}% <br>
                         <b>label4</b>: ${classes[3].label}, probability: ${classes[3].prob}% <br>
                         <b>label5</b>: ${classes[4].label}, probability: ${classes[4].prob}%`;
}

/*************************  Util.js  **************************/

// Load js script with async mode.
function asyncLoadScript(url, callback) {
  let script = document.createElement('script');

  script.async = true;
  script.type = 'text/javascript';
  script.onload = callback || function() {};
  script.src = url;
  document.getElementsByTagName('head')[0].appendChild(script);
}

// Get the model information from modelzoo.
function getModelByName(name) {
  for(const model of modelZoo) {
    if(name == model.name) {
      return model;
    }
  }
  return {};
}

// Load labels as String.
async function loadLabels(path) {
  let request = new Request(path);
  let response = await fetch(request);
  let lablesText = await response.text();
  let labels = lablesText.split('\n');

  return labels;
}

// Load model as ArrayBuffer.
async function loadModel(path) {
  let request = new Request(path);
  let response = await fetch(request);
  let modelArrayBuffer = await response.arrayBuffer();

  return modelArrayBuffer;
}

// Get image data array from imageElement.
function getImageData(imageElement, width, height, channelScheme='RGBA') {
  let imageData = [];
  const channels = 3;
  const imageChannels = 4; // RGBA
  const mean = [0, 0, 0, 0];
  const std = [1, 1, 1, 1];

  let canvas = document.createElement('canvas');
  canvas.setAttribute("width", width);
  canvas.setAttribute("height", height);
  let context = canvas.getContext('2d');
  context.drawImage(imageElement, 0, 0, width, height);
  let pixels = context.getImageData(0, 0, width, height).data;

  if(channelScheme === 'RGB') {
    // NHWC layout
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        for (let c = 0; c < channels; ++c) {
          let value = pixels[y*width*imageChannels + x*imageChannels + c];
          imageData[y*width*channels + x*channels + c] = (value - mean[c]) / std[c];
        }
      }
    }
  } else if(channelScheme === 'BGR') {
    // NCHW layout
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        for (let c = 0; c < channels; ++c) {
          let value = pixels[y*width*imageChannels + x*imageChannels + (channels-c-1)];
          imageData[y*width*channels + x*channels + c] = (value - mean[c]) / std[c];
        }
      }
    }
  } else if (channelScheme === 'RGBA') {
    // RAW layout
    imageData = [...pixels];
  } else {
    throw new Error(`Unknown color channel scheme ${channelScheme}`);
  }

  return imageData;
}

// Add softmax layer to gengerate the probility.
function softmax(arr) {
  const C = Math.max(...arr);
  const d = arr.map((y) => Math.exp(y - C)).reduce((a, b) => a + b);
  return arr.map((value, index) => { 
    return Math.exp(value - C) / d;
  })
}

// Find the top num labels from the forward result.
function getTopClasses(data, labels) {
  initdata = softmax(data);
  let probs = Array.from(initdata);
  let indexes = probs.map((prob, index) => [prob, index]);
  let sorted = indexes.sort((a, b) => {
    if (a[0] === b[0]) {return 0;}
    return a[0] < b[0] ? -1 : 1;
  });
  sorted.reverse();
  let classes = [];
  for (let i=0; i<topNum; ++i) {
    let prob = sorted[i][0];
    let index = sorted[i][1];
    let c = {
      label: labels[index],
      prob: (prob * 100).toFixed(2)
    }
    classes.push(c);
  }
  return classes;
}

// Print each inference result in the console.
function printResult(classes) {
  for (let i = 0; i < topNum; ++i){
    console.log(`label: ${classes[i].label}, probability: ${classes[i].prob}%`)
  }
}

// Generate the summarize result from the collect time.
function summarize(results) {
  if(results.length !== 0) {
    // remove first run, which is regarded as "warming up" execution
    results.shift();
    let d = results.reduce((d, v) => {
      d.sum += v;
      d.sum2 += v * v;
      return d;
    }, {
      sum: 0,
      sum2: 0
    });
    let mean = d.sum / results.length;
    let std = Math.sqrt((d.sum2 - results.length * mean * mean) / (results.length - 1));
    return {
      mean: mean,
      std: std
    };
  } else {
    return null;
  };
}
