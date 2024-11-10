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

import { SAM2 } from "./SAM2"

import { 
  LoaderCircle, 
} from 'lucide-react'

export default function Home() {
  const sam = useRef(null)
  // const [sentiment, setSentiment] = useState("Unknown")
  // const [text, setText] = useState("")

  const embedImage = async () => {
    await sam.current.embedImage()

  }

  useEffect(() => {
    if (!sam.current) {
      sam.current = new SAM2()
    }
  }, []);

  // useEffect(() => {
  //   if (bert.current) {
  //     predict(text)
  //   }
  // }, [text]);

  return (
    <div className="flex items-center justify-center min-h-screen bg-background p-4">
      <Card className="w-full max-w-2xl">
        <CardHeader>
          <CardTitle>SAM2</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Button onClick={embedImage}>Embed image</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
