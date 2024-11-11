import * as ort from 'onnxruntime-web';
import { AutoTokenizer } from '@huggingface/transformers';

import { Tensor } from 'onnxruntime-web';

export class SAM2 {
  static session = null
  image_encoded = null

  static async getSession() {
    if (!this.session) {
      this.session = await ort.InferenceSession.create('/onnx/sam2_hiera_tiny_encoder.with_runtime_opt.ort')
      console.log('onnxruntime-web inference session created')  
    }

    return this.session
  }

  constructor() {

  }

  async waitForSession() {
    await SAM2.getSession()
  }

  async encodeImage(inputTensor) {
    console.log("embedImage")
    const session = await SAM2.getSession()
    const results = await session.run({image: inputTensor});

    this.image_encoded = {
      high_res_feats_0: results[session.outputNames[0]],
      high_res_feats_1: results[session.outputNames[1]],
      image_embed: results[session.outputNames[2]]
    }
    console.log("embedImage: done")
  }

  async decode(point) {
    const session = await ort.InferenceSession.create(
        '/onnx/sam2_hiera_tiny_decoder.onnx'
    )

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

    const results = await session.run(inputs);

    return results
  }
}
