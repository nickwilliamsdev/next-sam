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

import { DistilbertSST } from "./Distilbert"

import { 
  LoaderCircle, 
} from 'lucide-react'

export default function Home() {
  const bert = useRef(null)
  const [sentiment, setSentiment] = useState("Unknown")
  const [text, setText] = useState("")

  const predict = async (text) => {
    const [isNegative, isPositive] = await bert.current.generate(text)

    if (isPositive > isNegative) {
      setSentiment("Positive (" + isPositive.toFixed(2) + ")")
    } else {
      setSentiment("Negative (" + isNegative.toFixed(2) + ")")
    }
  }

  useEffect(() => {
    if (!bert.current) {
      bert.current = new DistilbertSST()
    }
  }, []);

  useEffect(() => {
    if (bert.current) {
      predict(text)
    }
  }, [text]);

  return (
    <div className="flex items-center justify-center min-h-screen bg-background p-4">
      <Card className="w-full max-w-2xl">
        <CardHeader>
          <CardTitle>Next/distilbert-sst</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Textarea
              className="w-full"
              rows={4}
              value={text}
              placeholder="Analyze this!"
              onChange={(e) => setText(e.target.value)}
            />
            <p className="text-sm font-medium">Sentiment: {sentiment}</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
