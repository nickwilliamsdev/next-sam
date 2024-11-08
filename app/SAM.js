import { SamModel, AutoProcessor, RawImage, Tensor } from '@huggingface/transformers';

class SegmentAnythingSingleton {
    static model_id = 'Xenova/slimsam-77-uniform';
    static model;
    static processor;
    static quantized = true;

    static getInstance() {
        if (!this.model) {
            console.log("loading model ..")
            this.model = SamModel.from_pretrained(this.model_id, {
                quantized: this.quantized,
            });
            console.log("done")
        }
        if (!this.processor) {
            console.log("loading processor ..")
            this.processor = AutoProcessor.from_pretrained(this.model_id);
            console.log("done")
        }

        return Promise.all([this.model, this.processor]);
    }

    

}

export class SAM {
  image_embeddings = null
  image_inputs = null

  async loadModel(loadedCallback) {
    const [model, processor] = await SegmentAnythingSingleton.getInstance();

    loadedCallback()
  }

  async segmentImage(dataURL) {
    const [model, processor] = await SegmentAnythingSingleton.getInstance();
    const image = await RawImage.read(dataURL);

    this.image_inputs = await processor(image);
    this.image_embeddings = await model.get_image_embeddings(this.image_inputs)
  }

  async decodeMask(data) {
    // const image_inputs = images[imageid][0];
    // const image_embeddings = images[imageid][1]
    const [model, processor] = await SegmentAnythingSingleton.getInstance();

    const reshaped = this.image_inputs.reshaped_input_sizes[0];
    const points = data.map(x => [x.point[0] * reshaped[1], x.point[1] * reshaped[0]])
    const labels = data.map(x => BigInt(x.label));

    const input_points = new Tensor(
        'float32',
        points.flat(Infinity),
        [1, 1, points.length, 2],
    )
    const input_labels = new Tensor(
        'int64',
        labels.flat(Infinity),
        [1, 1, labels.length],
    )

    // Generate the mask
    const outputs = await model({
        ...this.image_embeddings,
        input_points,
        input_labels,
    })

    // Post-process the mask
    const scores = outputs.iou_scores.data
    let masks = await processor.post_process_masks(
        outputs.pred_masks,
        this.image_inputs.original_sizes,
        this.image_inputs.reshaped_input_sizes,
    );
    masks = RawImage.fromTensor(masks[0][0])

    return this.selectBestMask(masks, scores)
  }

  selectBestMask(mask, scores) {
    const maskCanvas = document.createElement('canvas');

    // Update canvas dimensions (if different)
    if (maskCanvas.width !== mask.width || maskCanvas.height !== mask.height) {
        maskCanvas.width = mask.width;
        maskCanvas.height = mask.height;
    }

    // Create context and allocate buffer for pixel data
    const context = maskCanvas.getContext('2d');
    const imageData = context.createImageData(maskCanvas.width, maskCanvas.height);

    // Select best mask
    const numMasks = scores.length; // 3
    let bestIndex = 0;
    for (let i = 1; i < numMasks; ++i) {
        if (scores[i] > scores[bestIndex]) {
            bestIndex = i;
        }
    }

    // Fill mask with colour
    const pixelData = imageData.data;
    for (let i = 0; i < pixelData.length; ++i) {
        const offset = 4 * i;
        if (mask.data[numMasks * i + bestIndex] === 1) {
            pixelData[offset] = 237;       // red
            pixelData[offset + 1] = 25; // green
            pixelData[offset + 2] = 233; // blue
            pixelData[offset + 3] = 100; // alpha
        }
    }

    // Draw image data to context
    context.putImageData(imageData, 0, 0);

    return maskCanvas
  }
}
