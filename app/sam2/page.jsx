"use client"

import React, { useState, useEffect, useRef, createContext } from 'react';

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

import { getImageData, canvasToTensor, resizeCanvas } from "@/lib/imageutils"

import { SAM2 } from "./SAM2"

import { 
  LoaderCircle, 
} from 'lucide-react'

export default function Home() {
  const sam = useRef(null)
  const canvasEl = useRef(null)
  const [imageURL, setImageURL] = useState("/photo.png")

  const embedImage = async () => {
    const canvas = canvasEl.current
    const imgTensor = canvasToTensor(resizeCanvas(canvas, {w: 1024, h: 1024}))
    await sam.current.embedImage(imgTensor)

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
            <canvas ref={canvasEl}/>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
