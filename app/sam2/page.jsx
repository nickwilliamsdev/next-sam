"use client"

import React, { useState, useEffect, useRef, createContext } from 'react';

import { Tensor } from 'onnxruntime-web';
import { getImageData, canvasToTensor, resizeCanvas, sliceTensorMask } from "@/lib/imageutils"
import { SAM2 } from "./SAM2"

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { LoaderCircle } from 'lucide-react'


export default function Home() {
  const sam = useRef(null)
  const canvasEl = useRef(null)
  const [imageURL, setImageURL] = useState("/photo.png")

  const embedImage = async () => {
    const canvas = canvasEl.current
    const imgTensor = canvasToTensor(resizeCanvas(canvas, {w: 1024, h: 1024}))
    await sam.current.embedImage(imgTensor)
  }

  const decodeMask = async (point) => {
    const decodingResults = await sam.current.decode(point) // decodingResults = [B=1, Masks, W, H]
    const maskTensor = decodingResults.masks
    const maskCanvas = sliceTensorMask(maskTensor, 0)    

    const targetCanvas = canvasEl.current
    const maskCanvasResized = resizeCanvas(maskCanvas, {w: targetCanvas.width, h: targetCanvas.height})

    targetCanvas.getContext('2d').drawImage(maskCanvasResized, 0, 0);
  }

  const imageClick = (event) => {
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


  useEffect(() => {
    if (!sam.current) {
      sam.current = new SAM2()
    }
  }, []);

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
          <CardTitle>SAM2</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Button onClick={embedImage}>Embed image</Button>
            <Button onClick={decodeMask}>Decode</Button>
            <canvas ref={canvasEl} onClick={imageClick}/>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
