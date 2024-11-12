"use client"

import React, { useState, useEffect, useRef, createContext } from 'react';

import { SAM } from './SlimSAM.js';

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"

import { 
  LoaderCircle, 
} from 'lucide-react'

export default function Home() {
  const [status, setStatus] = useState({loading: false, text: "Start"})
  const samWorker = useRef(null);
  const canvasEl = useRef(null)
  const [imageURL, setImageURL] = useState(null)

  const onSAMLoaded = () => {
    setStatus({loading: false, text: "Model ready"})
  }

  const segmentImage = async () => {
    if (canvasEl.current) {
      setStatus({loading: true, text: "Segmenting image"})

      const imgDataURL = canvasEl.current.toDataURL('image/png')
      await samWorker.current.segmentImage(imgDataURL)

      setStatus({loading: false, text: "Image Segmented"})
    } else {
      alert("No image loaded")
    }
  }

  const decodeMask = async (pos) => {
    setStatus({loading: true, text: "Decoding image"})

    const mask = await samWorker.current.decodeMask([pos])

    canvasEl.current.getContext('2d').drawImage(mask, 0, 0);

    setStatus({loading: false, text: "Image decoded"})
  }

  useEffect(() => {
    if (!samWorker.current) {
      samWorker.current = new SAM()
      samWorker.current.loadModel(onSAMLoaded)

      setStatus({loading: true, text: "Loading model"})
    }

    if (!imageURL) {
      setImageURL("/photo.png")
    }
  });

  useEffect(() => {
    if (imageURL) {
      const img = new Image();
      img.src = imageURL
      img.onload = function() {
        const canvas = canvasEl.current
        var ctx = canvas.getContext('2d');
        canvas.width = img.width
        canvas.height = img.height
        ctx.drawImage(img, 0, 0);
      }
    }
  }, [imageURL]);

  const imageClick = async (event) => {
    const canvas = canvasEl.current
    const rect = event.target.getBoundingClientRect();
    const pos = {
      point: [
        (event.clientX - rect.left) / canvas.width,
        (event.clientY - rect.top) / canvas.height
      ],
      label: 1
    }

    await decodeMask(pos)

  }

  return (
    <div className="grid grid-col gap-20 items-center justify-center h-screen">
      <div className="flex p-2">
      <Card>
        <CardHeader>
          <CardTitle>Next/SlimSAM (w/o web worker)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex">
            { status.loading ? <LoaderCircle className="animate-spin w-6 h-6" /> : "" }
            <p>{status.text}</p>
          </div>
          <Button onClick={segmentImage}>Segment</Button>
        </CardContent>
      </Card>
      </div>

      <canvas ref={canvasEl} onClick={imageClick} />
    </div>
  );
}
