import path from 'path';
import * as ort from 'onnxruntime-web';
import { Tensor } from 'onnxruntime-web';

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
      buffer = await fetch(url).then(response => response.arrayBuffer());
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
    await Promise.all([this.getEncoderSession(), this.getDecoderSession()])
  }

  async getEncoderSession() {
    if (!this.sessionEncoder) this.sessionEncoder = await ort.InferenceSession.create(this.bufferEncoder)

    return this.sessionEncoder
  }

  async getDecoderSession() {
    if (!this.sessionDecoder) this.sessionDecoder = await ort.InferenceSession.create(this.bufferDecoder)

    return this.sessionDecoder
  }

  async encodeImage(inputTensor) {
    const session = await this.getEncoderSession()
    const results = await session.run({image: inputTensor});

    this.image_encoded = {
      high_res_feats_0: results[session.outputNames[0]],
      high_res_feats_1: results[session.outputNames[1]],
      image_embed: results[session.outputNames[2]]
    }
  }

  async decode(point) {
    const session = await this.getDecoderSession()

    const inputs = {
      image_embed: this.image_encoded.image_embed, 
      high_res_feats_0: this.image_encoded.high_res_feats_0, 
      high_res_feats_1: this.image_encoded.high_res_feats_1,
      point_coords: new Tensor("float32", [point.x, point.y], [1, 1, 2]), 
      point_labels: new Tensor("float32", [point.label], [1, 1]), 
      mask_input: new Tensor("float32", new Float32Array(256 * 256), [1, 1, 256, 256]), 
      has_mask_input: new Tensor("float32", [0], [1]), 
      orig_im_size: new Tensor("int32", [1024, 1024], [2])
    }

    return await session.run(inputs);
  }
}
