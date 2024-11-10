
import { Tensor } from 'onnxruntime-web';

export function getImageData(canvas) {
  const ctx = canvas.getContext("2d")
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

  return imageData
}

export function resizeCanvas(canvasOrig, size) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')
    canvas.height = size.h
    canvas.width = size.w

    ctx.drawImage(canvasOrig, 0, 0, canvasOrig.width, canvasOrig.height, 0, 0, canvas.width, canvas.height)

    return canvas
  }

// inspired by: https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html
export function canvasToTensor(canvas) {
  const imageData = getImageData(canvas).data
  const dims = [1, 3, canvas.width, canvas.height]

  const [redArray, greenArray, blueArray] = [ [], [], [] ]

  for (let i = 0; i < imageData.length; i += 4) {
    redArray.push(imageData[i]);
    greenArray.push(imageData[i + 1]);
    blueArray.push(imageData[i + 2]);
    // skip data[i + 3] to filter out the alpha channel
  }

  const transposedData = redArray.concat(greenArray).concat(blueArray);

  let i, l = transposedData.length; // length, we need this for the loop
  // create the Float32Array size 3 * 224 * 224 for these dimensions output
  const float32Data = new Float32Array(dims[1] * dims[2] * dims[3]);
  for (i = 0; i < l; i++) {
    float32Data[i] = transposedData[i] / 255.0; // convert to float
  }

  return new Tensor("float32", float32Data, dims);
}