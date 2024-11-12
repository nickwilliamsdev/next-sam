"use client"

import * as ort from 'onnxruntime-web';
import { AutoTokenizer } from '@huggingface/transformers';

export class DistilbertSST {
  static session = null

  static async getSession() {
    if (!this.session) {
      this.session = await ort.InferenceSession.create('/onnx/distilbert-sst.onnx')
      console.log('Inference session created')            
    }

    return this.session
  }

  constructor() {
    DistilbertSST.getSession()
  }

  async generate(input_text) {
    const session = await DistilbertSST.getSession()

    const tokenizer = await AutoTokenizer.from_pretrained('Xenova/distilbert-base-uncased-finetuned-sst-2-english');
    const { input_ids } = await tokenizer(input_text);

    const results = await session.run({input: input_ids.ort_tensor});
    const outputName = session.outputNames[0]
    const output = results[outputName]

    // w/o HF tokenizer
    // const dataA = BigInt64Array.from([1n, 2n, 3n, 4n, 5n, 6n, 7n]);
    // const tensorA = new ort.Tensor('int64', dataA, [1, 7]);
    // const results = await session.run({input: tensorA});
    // console.log('Inference done')    

    return output.data       
  }
}
