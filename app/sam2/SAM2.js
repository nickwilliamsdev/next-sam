import * as ort from 'onnxruntime-web';
import { AutoTokenizer } from '@huggingface/transformers';

import { Tensor } from 'onnxruntime-web';


export class SAM2 {
  static session = null

  image_encoded = null

  static async getSession() {
    if (!this.session) {
      this.session = await ort.InferenceSession.create(
        '/onnx/sam2_hiera_tiny_encoder.with_runtime_opt.ort'
      )

      console.log('Inference session created')  
      console.log(this.session)
    }

    const tens = new Tensor('int32', [1, 2, 3, 4, 5, 6], [2, 3])

    return this.session
  }

  constructor() {
    SAM2.getSession()
  }

  async embedImage(inputTensor) {
    console.log("embedImage")
    const session = await SAM2.getSession()
    const dims = [1, 3, 1024, 1024]

    const results = await session.run({image: inputTensor});

    this.image_encoded = {
      high_res_feats_0: results[session.outputNames[0]],
      high_res_feats_1: results[session.outputNames[1]],
      image_embed: results[session.outputNames[2]]
    }
    console.log("embedImage: done")
  }

  async decode(pos) {
    const session = await ort.InferenceSession.create(
        '/onnx/sam2_hiera_tiny_decoder.onnx'
    )

    const inputs = {
      image_embed: this.image_encoded.image_embed, 
      high_res_feats_0: this.image_encoded.high_res_feats_0, 
      high_res_feats_1: this.image_encoded.high_res_feats_1,
      point_coords: new Tensor("float32", [pos.point.x, pos.point.y], [1, 1, 2]), 
      point_labels: new Tensor("float32", [1], [1, 1]), 
      mask_input: new Tensor("float32", new Float32Array(256 * 256), [1, 1, 256, 256]), 
      has_mask_input: new Tensor("float32", [0], [1]), 
      orig_im_size: new Tensor("int32", [1024, 1024], [2])
    }

    const results = await session.run(inputs);

    return results
  }
}
