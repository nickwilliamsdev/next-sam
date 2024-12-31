import path from 'path';

import * as ort from 'onnxruntime-web/all';
// ort.env.wasm.numThreads=1
// ort.env.wasm.simd = false;


const ENCODER_URL = "https://huggingface.co/g-ronimo/sam2-tiny/resolve/main/sam2_hiera_tiny_encoder.with_runtime_opt.ort"
const DECODER_URL = "https://huggingface.co/g-ronimo/sam2-tiny/resolve/main/sam2_hiera_tiny_decoder.onnx"

export class SAM2 {
  bufferEncoder = null
  bufferDecoder = null
  sessionEncoder = null
  sessionDecoder = null
  image_encoded = null

  constructor() { }

  async downloadModels() {
    this.bufferEncoder = await this.downloadModel(ENCODER_URL)
    this.bufferDecoder = await this.downloadModel(DECODER_URL)
  }

  async downloadModel(url) {
    // step 1: check if cached
    const root = await navigator.storage.getDirectory();
    const filename = path.basename(url);

    let fileHandle = await root.getFileHandle(filename).catch(e => console.error("File does not exist:", filename, e));

    if (fileHandle) {
      const file = await fileHandle.getFile();
      if (file.size>0) return await file.arrayBuffer()
    }

    // step 2: download if not cached
    console.log("File " + filename + " not in cache, downloading from " + url)
    let buffer = null
    try {
      buffer = await fetch(
        url,
        {
          headers: new Headers({
              Origin: location.origin,
          }),
          mode: 'cors',
        }).then(response => response.arrayBuffer());
    }
    catch (e) {
      console.error("Download of " + url + " failed: ", e)
      return null
    }

    // step 3: store 
    try {
      const fileHandle = await root.getFileHandle(filename, { create: true });
      const writable = await fileHandle.createWritable();
      await writable.write(buffer);
      await writable.close();

      console.log("Stored " + filename)
    }
    catch (e) {
      console.error("Storage of " + filename + " failed: ", e)
    }
    return buffer
  }

  async createSessions() {
    const success = await this.getEncoderSession() && await this.getDecoderSession()
    
    return {
      success: success,
      device: success ? this.sessionEncoder[1] : null
    }
  }

  async getORTSession(model) {
    /** Creating a session with executionProviders: {"webgpu", "cpu"} fails
     *  => "Error: multiple calls to 'initWasm()' detected."
     *  but ONLY in Safari and Firefox (wtf)
     *  seems to be related to web worker, see https://github.com/microsoft/onnxruntime/issues/22113
     *  => loop through each ep, catch e if not available and move on
     */
    let session = null
    for (let ep of ["webgpu", "cpu"]) {
      try { session = await ort.InferenceSession.create(model, { executionProviders: [ep]}) }
      catch (e) { console.error(e); continue }

      return [session, ep]
    }
  }

  async getEncoderSession() {
    if (!this.sessionEncoder) this.sessionEncoder = await this.getORTSession(this.bufferEncoder)

    return this.sessionEncoder
  }

  async getDecoderSession() {
    if (!this.sessionDecoder) this.sessionDecoder = await this.getORTSession(this.bufferDecoder)

    return this.sessionDecoder
  }

  async encodeImage(inputTensor) {
    const [session, device] = await this.getEncoderSession()
    const results = await session.run({image: inputTensor});

    this.image_encoded = {
      high_res_feats_0: results[session.outputNames[0]],
      high_res_feats_1: results[session.outputNames[1]],
      image_embed: results[session.outputNames[2]]
    }
  }

  async decode(point) {
    const [session, device] = await this.getDecoderSession()

    const inputs = {
      image_embed: this.image_encoded.image_embed, 
      high_res_feats_0: this.image_encoded.high_res_feats_0, 
      high_res_feats_1: this.image_encoded.high_res_feats_1,
      point_coords: new ort.Tensor("float32", [point.x, point.y], [1, 1, 2]), 
      point_labels: new ort.Tensor("float32", [point.label], [1, 1]), 
      mask_input: new ort.Tensor("float32", new Float32Array(256 * 256), [1, 1, 256, 256]), 
      has_mask_input: new ort.Tensor("float32", [0], [1]), 
      orig_im_size: new ort.Tensor("int32", [1024, 1024], [2])
    }

    return await session.run(inputs);
  }

  async decodeFull() {
    const [session, device] = await this.getDecoderSession()

    const inputs = {
      image_embed: this.image_encoded.image_embed, 
      high_res_feats_0: this.image_encoded.high_res_feats_0, 
      high_res_feats_1: this.image_encoded.high_res_feats_1,
      point_coords: new ort.Tensor("float32", [], []), 
      point_labels: new ort.Tensor("float32", [], []), 
      mask_input: new ort.Tensor("float32", new Float32Array(256 * 256), [1, 1, 256, 256]), 
      has_mask_input: new ort.Tensor("float32", [0], [1]), 
      orig_im_size: new ort.Tensor("int32", [1024, 1024], [2])
    }

    return await session.run(inputs);
  }

  generateCropBoxes(imageWidth, imageHeight, layers, overlapRatio) {
    const cropBoxes = [];
    const layerScales = Array.from({ length: layers }, (_, i) => Math.pow(2, i));

    for (const scale of layerScales) {
        const cropWidth = imageWidth / scale;
        const cropHeight = imageHeight / scale;
        const stepX = cropWidth * (1 - overlapRatio);
        const stepY = cropHeight * (1 - overlapRatio);

        for (let y = 0; y < imageHeight; y += stepY) {
            for (let x = 0; x < imageWidth; x += stepX) {
                const x0 = Math.floor(x);
                const y0 = Math.floor(y);
                const x1 = Math.min(imageWidth, Math.ceil(x + cropWidth));
                const y1 = Math.min(imageHeight, Math.ceil(y + cropHeight));
                cropBoxes.push([x0, y0, x1, y1]);
            }
        }
    }
    return cropBoxes;
  }
  batchProcess(points, batchSize) {
    const batches = [];
    for (let i = 0; i < points.length; i += batchSize) {
        batches.push(points.slice(i, i + batchSize));
    }
    return batches;
  }
  // Generate automatic prompts (e.g., grid points)
  generatePrompts(height, width, gridSpacing = 16) {
    const prompts = [];
    for (let y = 0; y < height; y += gridSpacing) {
        for (let x = 0; x < width; x += gridSpacing) {
            prompts.push([y, x, 1]); // (y, x, is_positive)
        }
    }
    return new Float32Array(prompts.flat());
  }
}
