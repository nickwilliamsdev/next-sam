"use client"

import React, { useState, useEffect, useRef, createContext } from 'react';

import { Tensor } from 'onnxruntime-web';
import { getImageData, canvasToFloat32Array, resizeCanvas, sliceTensorMask } from "@/lib/imageutils"
// import { SAM2 } from "./SAM2"

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { LoaderCircle } from 'lucide-react'


export default function Home() {
  // const sam = useRef(null)
  const [loading, setLoading] = useState(false)
  const [samWorkerReady, setSamWorkerReady] = useState(false)
  const [imageEncoded, setImageEncoded] = useState(false)

  const samWorker = useRef(null)
  const canvasEl = useRef(null)
  const [imageURL, setImageURL] = useState("/photo.png")

  const encodeImage = async () => {
    const canvas = canvasEl.current
    const float32Data = canvasToFloat32Array(resizeCanvas(canvas, {w: 1024, h: 1024}))

    samWorker.current.postMessage({ 
      type: 'encodeImage',
      data: float32Data
    });   
    setLoading(true)
  }

  const decodeMask = async (point) => {
    samWorker.current.postMessage({ 
      type: 'decodeMask',
      data: point
    });   
  }

  const drawMask = (decodingResults) => {
    // SAM2 returns 3 mask along with scores -> select best one    
    const maskTensors = decodingResults.masks
    const maskScores = decodingResults.iou_predictions.cpuData
    const bestMaskIdx = maskScores.indexOf(Math.max(...maskScores))
    const maskCanvas = sliceTensorMask(maskTensors, bestMaskIdx)    

    // draw mask on top of input image
    const targetCanvas = canvasEl.current
    const maskCanvasResized = resizeCanvas(maskCanvas, {w: targetCanvas.width, h: targetCanvas.height})
    targetCanvas.getContext('2d').drawImage(maskCanvasResized, 0, 0);
  }

  const imageClick = (event) => {
    if (!imageEncoded) {
      return
    }

    const canvas = canvasEl.current
    const rect = event.target.getBoundingClientRect();

    // input image will be resized to 1024x1024 -> also normalize pos to 1024x1024
    const point = {
      x: (event.clientX - rect.left) / canvas.width * 1024,
      y: (event.clientY - rect.top) / canvas.height * 1024,
      label: 1
    }

    decodeMask(point)
  }

  const onWorkerMessage = (event) => {
    const {type, data} = event.data

    // console.log("Main thread onWorkerMessage")
    // console.log(event.data)

    if (type == "pong" ) {
      setLoading(false)
      setSamWorkerReady(true)
    } else if (type == "encodeImageDone" ) {
      setLoading(false)
      setImageEncoded(true)
    } else if (type == "decodeMaskResult" ) {
      drawMask(data) 
    }
  }


  useEffect(() => {
    if (!samWorker.current) {
      samWorker.current = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
      samWorker.current.addEventListener('message', onWorkerMessage)
      samWorker.current.postMessage({ type: 'ping' });   

      setLoading(true)
    }
  })

  useEffect(() => {
    if (imageURL) {
      const img = new Image();
      img.src = imageURL
      img.onload = function() {
        const canvas = canvasEl.current
        var ctx = canvas.getContext('2d');
        canvas.width = 512
        canvas.height = 512
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height );
      }
    }
  }, [imageURL]);

  return (
    <div className="flex items-center justify-center min-h-screen bg-background p-4">
      <Card className="w-full max-w-2xl">
        <CardHeader>
          <CardTitle>Next/SAM2 - Image Segmentation in the browser with onnxruntime-web and Meta's SAM2</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Button 
              onClick={encodeImage}
              disabled={loading || (samWorkerReady && imageEncoded)}
            >
              <p className="flex items-center gap-2">
              { loading &&
                  <LoaderCircle className="animate-spin w-6 h-6" />
              }
              { loading && !samWorkerReady && "Loading model"}
              { !loading && samWorkerReady && !imageEncoded && "Encode image"}
              { loading && samWorkerReady && !imageEncoded && "Encoding"}
              { !loading && samWorkerReady && imageEncoded && "Ready. Click on image"}
              </p>
            </Button>
            <canvas ref={canvasEl} onClick={imageClick}/>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
