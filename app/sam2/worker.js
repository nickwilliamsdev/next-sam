import { SAM2 } from "./SAM2"
import { Tensor } from 'onnxruntime-web';

const sam = new SAM2()

self.onmessage = async (e) => {
    console.log("worker received message")

    const { type, data } = e.data;

    if (type === 'ping') {
        await sam.waitForSession()
        self.postMessage({ type: 'pong' })

    } else if (type === 'encodeImage') {
        const {float32Array, shape} = data
        const imgTensor = new Tensor("float32", float32Array, shape);

        await sam.encodeImage(imgTensor)

        self.postMessage({ type: 'encodeImageDone' })

    } else if (type === 'decodeMask') {
        const point = data
        const decodingResults = await sam.decode(point) // decodingResults = Tensor [B=1, Masks, W, H]

        self.postMessage({ 
            type: 'decodeMaskResult',
            data: decodingResults
        })

    } else {
        throw new Error(`Unknown message type: ${type}`);
    }
}
