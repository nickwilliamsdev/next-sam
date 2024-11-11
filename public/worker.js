import { SamModel, AutoProcessor, RawImage, Tensor } from '@huggingface/transformers';


// if (!samWorker.current) {
//   samWorker.current = new Worker(new URL('../public/worker.js', import.meta.url), { type: 'module' });

//   setLoading(true)
//   setStatus("Worker init")

//   samWorker.current.postMessage({ type: "ping" });
// }

// samWorker.current.addEventListener('message', onWorkerMessageReceived);

// return () => samWorker.current.removeEventListener('message', onWorkerMessageReceived);

export class SegmentAnythingSingleton {
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

// key/value store for embeddings, inputs, ready
let images = { }

self.onmessage = async (e) => {
    console.log("worker received message")

    const [model, processor] = await SegmentAnythingSingleton.getInstance();

    const { imageid, type, data } = e.data;
    if (type === 'ping') {
        self.postMessage({type: 'pong'});
        
    } else if (type === 'segment') {
        const image = await RawImage.read(data);

        let image_inputs = await processor(image);
        let image_embeddings = await model.get_image_embeddings(image_inputs)

        images[imageid] = [image_inputs, image_embeddings] 

        self.postMessage({
            imageid: imageid,
            type: 'segment_result',
            data: 'done',
        });

    } else if (type === 'decode') {
        const image_inputs = images[imageid][0];
        const image_embeddings = images[imageid][1]

        const reshaped = image_inputs.reshaped_input_sizes[0];
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
            ...image_embeddings,
            input_points,
            input_labels,
        })

        // Post-process the mask
        const masks = await processor.post_process_masks(
            outputs.pred_masks,
            image_inputs.original_sizes,
            image_inputs.reshaped_input_sizes,
        );
        const scores = outputs.iou_scores.data

        // Send the result back to the main thread
        self.postMessage({
            imageid: imageid,
            type: 'decode_result',
            data: {
                masks: RawImage.fromTensor(masks[0][0]),
                scores: scores,
            }
        });
    } else {
        throw new Error(`Unknown message type: ${type}`);
    }
}
