
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

export function sliceTensorMask(maskTensor, maskIdx) {
  const [bs, noMasks, width, height] = maskTensor.dims
  const stride = width * height
  const start = stride * maskIdx, end = start + stride

  const maskData = maskTensor.cpuData.slice(start, end);

  const C = 4 // 4 channels, RGBA
  const imageData = new Uint8ClampedArray(stride * C)
  for (let srcIdx = 0; srcIdx < maskData.length; srcIdx++) {
    const trgIdx = srcIdx * C
    const maskedPx = maskData[srcIdx] > 0
    imageData[trgIdx] = maskedPx > 0 ? 255 : 0
    imageData[trgIdx + 1] = 0
    imageData[trgIdx + 2] = 0
    imageData[trgIdx + 3] = maskedPx > 0 ? 150 : 0 // alpha
  }

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')
  canvas.height = height
  canvas.width = width
  ctx.putImageData(new ImageData(imageData, width, height), 0, 0);

  return canvas
}

// inspired by: https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html
export function canvasToFloat32Array(canvas) {
  const imageData = getImageData(canvas).data
  const shape = [1, 3, canvas.width, canvas.height]

  const [redArray, greenArray, blueArray] = [ [], [], [] ]

  for (let i = 0; i < imageData.length; i += 4) {
    redArray.push(imageData[i]);
    greenArray.push(imageData[i + 1]);
    blueArray.push(imageData[i + 2]);
    // skip data[i + 3] to filter out the alpha channel
  }

  const transposedData = redArray.concat(greenArray).concat(blueArray);

  let i, l = transposedData.length; 
  const float32Array = new Float32Array(shape[1] * shape[2] * shape[3]);
  for (i = 0; i < l; i++) {
    float32Array[i] = transposedData[i] / 255.0; // convert to float
  }

  return {float32Array, shape}
}



