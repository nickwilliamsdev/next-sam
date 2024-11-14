# Client-side image segmentation with SAM2
This is a Next.js application that performs image segmentation using Meta's Segment Anything Model (SAM2) and onnxruntime-web. The app allows users to encode an image, decode masks by clicking on the image, and crop the image using the decoded mask.

Demo at [sam2-seven.vercel.app](https://sam2-seven.vercel.app/)

https://github.com/user-attachments/assets/0d3b9f3b-2ab1-4627-9662-fca1a7cc2289

# Features
* Utilizes Meta's SAM2 model for segmentation
* onnxruntime-web for model inference
* webgpu with fallback to CPU if not available or not supported by browser
* Model storage using OPFS ([not working](https://bugs.webkit.org/show_bug.cgi?id=231706) in Safari)
* Image upload 
* Mask decoding based on point prompt
* Cropping

# Installation
Clone the repository:

```
git clone https://github.com/geronimi73/next-sam
cd next-sam
npm install
npm run dev
```

Open your browser and visit http://localhost:3000 

# Usage
1. Upload an image or use the default image.
2. Click the "Encode image" button to start encoding the image.
3. Once the encoding is complete, click on the image to decode masks.
4. Click the "Crop" button to crop the image using the decoded mask.

# Technologies Used
* Next.js
* Shadcn/ui components
* onnxruntime-web
* Meta's Segment Anything Model V2 (SAM2)

# License
This project is licensed under the MIT License.

# Acknowledgements
* Meta's Segment Anything Model (SAM2)
* onnxruntime-web
* Shadcn/ui components

Last but not least!
* [transformer.js](https://github.com/huggingface/transformers.js)
* https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything
* https://github.com/lucasgelfond/webgpu-sam2
* https://github.com/microsoft/onnxruntime-inference-examples
