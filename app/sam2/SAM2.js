
import * as ort from 'onnxruntime-web';
import { AutoTokenizer } from '@huggingface/transformers';

// import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.min.js";

export class SAM2 {
  static session = null
  static loading = false

  static async getSession() {
    console.log("getSession")
    if (!this.session) {
      this.session = await ort.InferenceSession.create('/onnx/sam2_hiera_tiny_encoder.with_runtime_opt.ort')

      console.log('Inference session created')  
      console.log(this.session)
    }

    return this.session
  }

  constructor() {
    SAM2.getSession()
  }

  async embedImage() {
    const session = await SAM2.getSession()

    // const tokenizer = await AutoTokenizer.from_pretrained('Xenova/distilbert-base-uncased-finetuned-sst-2-english');
    // const { input_ids } = await tokenizer(input_text);

    // const results = await session.run({input: input_ids.ort_tensor});
    // const outputName = session.outputNames[0]
    // const output = results[outputName]

    // // w/o HF tokenizer
    // // const dataA = BigInt64Array.from([1n, 2n, 3n, 4n, 5n, 6n, 7n]);
    // // const tensorA = new ort.Tensor('int64', dataA, [1, 7]);
    // // const results = await session.run({input: tensorA});
    // // console.log('Inference done')    

    // return output.data       
  }
}
