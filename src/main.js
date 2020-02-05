let testInfo = document.getElementById("testInfo");
let iterationProgress = document.getElementById("iterationProgress");
let showIterations = document.getElementById("showIterations");
let libraryName = document.getElementById("libName");
let backendName = document.getElementById("backendName");
let runButton = document.getElementById("runButton");
let imgElement = document.getElementById("imgsrc");
let inputElement = document.getElementById("fileinput");
let runtimeStatus = document.getElementById("runtimeStatus");
let modelStatus = document.getElementById("modelStatus");

libraryName.addEventListener("change", setLibrary);
backendName.addEventListener("change", setBackend);
inputElement.addEventListener("change", (e) => {
  imgElement.src = URL.createObjectURL(e.target.files[0]);
}, false);
runButton.addEventListener("click", run);

loadMLRuntime();

/*************************   CONTROL PARAMETERS   **************************/

// The forward iterations set by user
let iterations = Number(document.querySelector('#iterations').value);
// Number of top result to show
let topNum = 5;
// Save each forward time 
let timeSum = [];

// Detect the click, init the UI and control parameters, then run the test.
async function run() {
  let library = libraryName.value;
  let modelName = document.getElementById("modelName").value;
  let model = getModelByName(modelName);
  initPara();
  if(library == 'OpenCV.js') {
    cvPredict(model);
  } else if(library == 'ONNX.js') {
    onnxPredict(model);
  }
}

function setLibrary() {
  window.sessionStorage.clear();
  window.sessionStorage.setItem("library", libraryName.value);
  window.location.reload(true);
}

function setBackend() {
  window.sessionStorage.setItem("backend", backendName.value);
  window.location.reload(true);
}

// Async load ML Library
function loadMLRuntime() {
  let library = window.sessionStorage.getItem("library") || libraryName.value;
  let backend = window.sessionStorage.getItem("backend") || backendName.value;

  updateSelectBox(libraryName, library);
  updateSelectBox(backendName, backend);

  if(library == 'OpenCV.js') {
    asyncLoadScript(`./opencv.js/${backend}/opencv.js`, onOpenCvReady);
    setSelectOption(backendName, 'SIMD', 'show');
    setSelectOption(backendName, 'Threads+SIMD', 'show');
  } else if(library == 'ONNX.js') {
    asyncLoadScript('./onnx.js/onnx.min.js', onOnnxjsReady);
    setSelectOption(backendName, 'SIMD', 'hide');
    setSelectOption(backendName, 'Threads+SIMD', 'hide');
  } else {
    console.log('Undifinded ML library.');
  }
}

//Init the UI and the control parameters.
function initPara() {
  iterations = Number(document.querySelector('#iterations').value);
  timeSum = [];
  updatInnerHTML(testInfo, '');
  updateStatusPicture(modelStatus, 0);
  iterationProgress.style.visibility = 'hidden';
  updatInnerHTML(showIterations, '');
  runButton.disabled = true;
}
