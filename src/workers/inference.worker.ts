/// <reference lib="webworker" />

import { InferenceSession, Tensor } from 'onnxruntime-web'

declare const self: WorkerGlobalScope & typeof globalThis

const CONFIG = {
  numFrames: 20,
  inputSize: 100,
  yoloInputSize: 640,
  stride: 1,
  threshold: 0.7,
  minFrames: 20,
  yoloModelPath: '/models/yolo26n-pose.onnx',
  violenceModelPath: '/models/violence_detector_standalone.onnx',
  gamma: 0.67
}

const gammaTable = new Uint8Array(256)
for (let i = 0; i < 256; i++) {
  gammaTable[i] = Math.round(Math.pow(i / 255, CONFIG.gamma) * 255)
}

interface PoseKeypoint {
  x: number
  y: number
  confidence: number
}

interface InferenceResult {
  score: number
  isFight: boolean
  timestamp: number
}

interface WorkerMessage {
  type: 'init' | 'frame' | 'reset' | 'config'
  data?: any
}

interface WorkerResponse {
  type: 'ready' | 'result' | 'error' | 'buffer' | 'status'
  data: any
}

class RingBuffer<T> {
  private buffer: T[]
  private capacity: number
  private head = 0
  private count = 0

  constructor(capacity: number) {
    this.capacity = capacity
    this.buffer = new Array(capacity)
  }

  push(item: T): void {
    this.buffer[this.head] = item
    this.head = (this.head + 1) % this.capacity
    if (this.count < this.capacity) {
      this.count++
    }
  }

  get(index: number): T | undefined {
    if (index < 0 || index >= this.count) return undefined
    const idx = (this.head - this.count + index + this.capacity) % this.capacity
    return this.buffer[idx]
  }

  getAll(): T[] {
    const result: T[] = []
    for (let i = 0; i < this.count; i++) {
      const item = this.get(i)
      if (item) result.push(item)
    }
    return result
  }

  isFull(): boolean {
    return this.count >= this.capacity
  }

  getLength(): number {
    return this.count
  }

  clear(): void {
    this.head = 0
    this.count = 0
  }

  toArray(): T[] {
    return this.getAll()
  }
}

class ViolenceDetectionWorker {
  private yoloSession: InferenceSession | null = null
  private violenceSession: InferenceSession | null = null
  private poseBuffer: RingBuffer<Float32Array>
  private changeBuffer: RingBuffer<Float32Array>

  private poseCanvas: OffscreenCanvas
  private poseCtx: OffscreenCanvasRenderingContext2D

  private changeCanvas: OffscreenCanvas
  private changeCtx: OffscreenCanvasRenderingContext2D


  private lastFrame: Float32Array | null = null
  private frameCounter = 0
  private isProcessing = false
  private lastInferenceTime = 0
  private lastFrameProcessTime = 0
  private readonly frameInterval = 1000 / 20

  private poseSeqData: Float32Array
  private changeSeqData: Float32Array

  private keypointConnections = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [9, 12], [5, 6]
  ]

  constructor() {
    const totalElements = 1 * CONFIG.numFrames * 3 * CONFIG.inputSize * CONFIG.inputSize
    this.poseSeqData = new Float32Array(totalElements)
    this.changeSeqData = new Float32Array(totalElements)

    this.poseBuffer = new RingBuffer<Float32Array>(CONFIG.numFrames)
    this.changeBuffer = new RingBuffer<Float32Array>(CONFIG.numFrames)

    this.poseCanvas = new OffscreenCanvas(CONFIG.inputSize, CONFIG.inputSize)
    this.poseCtx = this.poseCanvas.getContext('2d', { willReadFrequently: true })!

    this.changeCanvas = new OffscreenCanvas(CONFIG.inputSize, CONFIG.inputSize)
    this.changeCtx = this.changeCanvas.getContext('2d', { willReadFrequently: true })!
  }

  async initialize(): Promise<void> {
    try {
      console.log('[Worker] Starting initialization...')

      const ort = await import('onnxruntime-web')
      console.log('[Worker] ONNX Runtime loaded')

      const executionProviders = ['wasm']

      try {
        console.log('[Worker] Loading violence detector model...')
        this.violenceSession = await ort.InferenceSession.create(CONFIG.violenceModelPath, {
          executionProviders,
          graphOptimizationLevel: 'all'
        })
        console.log('[Worker] Violence detector model loaded successfully')
      } catch (e) {
        console.error('[Worker] Failed to load violence model:', e)
        this.postMessage({ type: 'error', data: `Failed to load violence model: ${e}` })
        return
      }

      try {
        console.log('[Worker] Loading YOLO-Pose model...')
        this.yoloSession = await ort.InferenceSession.create(CONFIG.yoloModelPath, {
          executionProviders,
          graphOptimizationLevel: 'all'
        })
        console.log('[Worker] YOLO-Pose model loaded')
      } catch (e) {
        console.warn('[Worker] YOLO-Pose model not found, using fallback pose estimation')
        this.yoloSession = null
      }

      console.log('[Worker] Sending ready message...')
      this.postMessage({ type: 'ready', data: { yoloLoaded: !!this.yoloSession } })
      console.log('[Worker] Ready message sent')
    } catch (error) {
      console.error('[Worker] Initialization failed:', error)
      this.postMessage({ type: 'error', data: (error as Error).message })
    }
  }

  private postMessage(response: WorkerResponse): void {
    self.postMessage(response)
  }

  async processFrame(videoFrame: ImageBitmap): Promise<void> {
    const now = Date.now()

    if (now - this.lastFrameProcessTime < this.frameInterval) {
      return
    }

    if (this.isProcessing) {
      return
    }

    this.frameCounter++

    if (this.frameCounter % CONFIG.stride !== 0) {
      return
    }

    this.isProcessing = true
    this.lastFrameProcessTime = now

    try {
      const [poseData, changeData] = await Promise.all([
        this.extractPoseSequence(videoFrame),
        this.extractChangeSequence(videoFrame)
      ])

      this.poseBuffer.push(poseData)
      this.changeBuffer.push(changeData)

      this.postMessage({
        type: 'buffer',
        data: {
          poseLength: this.poseBuffer.getLength(),
          changeLength: this.changeBuffer.getLength(),
          progress: this.poseBuffer.getLength() / CONFIG.minFrames
        }
      })

      if (this.poseBuffer.isFull()) {
        await this.runInference()
      }
    } catch (error) {
      console.error('[Worker] Frame processing error:', error)
      this.postMessage({ type: 'error', data: (error as Error).message })
    } finally {
      this.isProcessing = false
    }
  }

  private async extractPoseSequence(_videoFrame: ImageBitmap): Promise<Float32Array> {
    this.poseCtx.fillStyle = 'black'
    this.poseCtx.fillRect(0, 0, CONFIG.inputSize, CONFIG.inputSize)

    let keypoints: PoseKeypoint[] = []

    if (this.yoloSession) {
      keypoints = await this.runYoloPose(_videoFrame)
    } else {
      keypoints = this.generateFallbackPose(_videoFrame)
    }

    this.drawSkeleton(keypoints)

    const imageData = this.poseCtx.getImageData(0, 0, CONFIG.inputSize, CONFIG.inputSize)
    const poseData = this.preprocessPoseImage(imageData)

    return poseData
  }

  private async runYoloPose(videoFrame: ImageBitmap): Promise<PoseKeypoint[]> {
    const inputTensor = this.videoFrameToTensor(videoFrame)

    try {
      const feeds: Record<string, Tensor> = {
        'images': inputTensor
      }

      const results = await this.yoloSession!.run(feeds)
      const output = results['output0']

      if (!output) {
        return this.generateFallbackPose(videoFrame)
      }

      return this.parseYoloOutput(output, videoFrame.width, videoFrame.height)
    } catch (error) {
      console.error('[Worker] YOLO inference error:', error)
      return this.generateFallbackPose(videoFrame)
    } finally {
      inputTensor.dispose()
    }
  }

  private videoFrameToTensor(videoFrame: ImageBitmap): Tensor {
    const canvas = new OffscreenCanvas(CONFIG.yoloInputSize, CONFIG.yoloInputSize)
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(videoFrame, 0, 0, CONFIG.yoloInputSize, CONFIG.yoloInputSize)

    const imageData = ctx.getImageData(0, 0, CONFIG.yoloInputSize, CONFIG.yoloInputSize)
    const { data } = imageData

    for (let i = 0; i < data.length; i += 4) {
      data[i] = gammaTable[data[i]]
      data[i + 1] = gammaTable[data[i + 1]]
      data[i + 2] = gammaTable[data[i + 2]]
    }

    const size = CONFIG.yoloInputSize
    const tensorData = new Float32Array(1 * 3 * size * size)

    for (let h = 0; h < size; h++) {
      for (let w = 0; w < size; w++) {
        const srcIdx = (h * size + w) * 4
        const dstIdx = h * size + w

        tensorData[dstIdx] = data[srcIdx + 0] / 255.0
        tensorData[size * size + dstIdx] = data[srcIdx + 1] / 255.0
        tensorData[2 * size * size + dstIdx] = data[srcIdx + 2] / 255.0
      }
    }

    return new Tensor('float32', tensorData, [1, 3, size, size])
  }

  private parseYoloOutput(output: Tensor, origWidth: number, origHeight: number): PoseKeypoint[] {
    const data = output.data as Float32Array
    const numKeypoints = 17

    const keypoints: PoseKeypoint[] = []

    for (let i = 0; i < numKeypoints; i++) {
      const baseIdx = i * 3

      const x = data[baseIdx] * origWidth
      const y = data[baseIdx + 1] * origHeight
      const confidence = data[baseIdx + 2]

      keypoints.push({ x, y, confidence })
    }

    return keypoints
  }

  private generateFallbackPose(videoFrame: ImageBitmap): PoseKeypoint[] {
    const w = videoFrame.width
    const h = videoFrame.height

    const centerX = w / 2
    const centerY = h / 2
    const scale = Math.min(w, h) * 0.3

    const keypoints: PoseKeypoint[] = [
      { x: centerX, y: centerY * 0.9, confidence: 0.9 },
      { x: centerX - scale * 0.2, y: centerY * 0.5, confidence: 0.8 },
      { x: centerX + scale * 0.2, y: centerY * 0.5, confidence: 0.8 },
      { x: centerX - scale * 0.3, y: centerY * 0.7, confidence: 0.7 },
      { x: centerX - scale * 0.4, y: centerY * 0.9, confidence: 0.6 },
      { x: centerX + scale * 0.2, y: centerY * 0.5, confidence: 0.8 },
      { x: centerX + scale * 0.3, y: centerY * 0.7, confidence: 0.7 },
      { x: centerX + scale * 0.4, y: centerY * 0.9, confidence: 0.6 },
      { x: centerX - scale * 0.1, y: centerY * 0.3, confidence: 0.85 },
      { x: centerX - scale * 0.2, y: centerY * 0.1, confidence: 0.75 },
      { x: centerX - scale * 0.25, y: centerY * -0.05, confidence: 0.65 },
      { x: centerX - scale * 0.3, y: centerY * -0.15, confidence: 0.55 },
      { x: centerX + scale * 0.1, y: centerY * 0.3, confidence: 0.85 },
      { x: centerX + scale * 0.2, y: centerY * 0.1, confidence: 0.75 },
      { x: centerX + scale * 0.25, y: centerY * -0.05, confidence: 0.65 },
      { x: centerX + scale * 0.3, y: centerY * -0.15, confidence: 0.55 },
      { x: centerX, y: centerY * 0.2, confidence: 0.8 }
    ]

    return keypoints
  }

  private drawSkeleton(keypoints: PoseKeypoint[]): void {
    if (keypoints.length === 0) return

    const scaleX = CONFIG.inputSize / 640
    const scaleY = CONFIG.inputSize / 640

    this.poseCtx.strokeStyle = 'white'
    this.poseCtx.lineWidth = 2
    this.poseCtx.lineCap = 'round'

    for (const [i, j] of this.keypointConnections) {
      if (keypoints[i] && keypoints[j]) {
        this.poseCtx.beginPath()
        this.poseCtx.moveTo(keypoints[i].x * scaleX, keypoints[i].y * scaleY)
        this.poseCtx.lineTo(keypoints[j].x * scaleX, keypoints[j].y * scaleY)
        this.poseCtx.stroke()
      }
    }

    this.poseCtx.fillStyle = 'white'
    for (const kp of keypoints) {
      const x = kp.x * scaleX
      const y = kp.y * scaleY
      this.poseCtx.beginPath()
      this.poseCtx.arc(x, y, 3, 0, Math.PI * 2)
      this.poseCtx.fill()
    }
  }

  private preprocessPoseImage(imageData: ImageData): Float32Array {
    const { width, height, data } = imageData
    const size = CONFIG.inputSize
    const result = new Float32Array(size * size * 3)
    const channelStride = size * size // ???????????????

    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        const srcIdx = (h * width + w) * 4
        const pixelIdx = h * width + w

        // ???_?? CHW ?x????
        result[pixelIdx] = data[srcIdx + 0] / 255.0                       // R ???
        result[channelStride + pixelIdx] = data[srcIdx + 1] / 255.0       // G ???
        result[channelStride * 2 + pixelIdx] = data[srcIdx + 2] / 255.0   // B ???
      }
    }
    return result
  }

  private async extractChangeSequence(videoFrame: ImageBitmap): Promise<Float32Array> {
    this.changeCtx.drawImage(videoFrame, 0, 0, CONFIG.inputSize, CONFIG.inputSize)
    const imageData = this.changeCtx.getImageData(0, 0, CONFIG.inputSize, CONFIG.inputSize)

    const { data } = imageData
    for (let i = 0; i < data.length; i += 4) {
      data[i] = gammaTable[data[i]]
      data[i + 1] = gammaTable[data[i + 1]]
      data[i + 2] = gammaTable[data[i + 2]]
    }

    const currentFrame = this.preprocessChangeImage(imageData)

    let changeData: Float32Array

    if (this.lastFrame) {
      changeData = this.computeFrameDifference(this.lastFrame, currentFrame)
    } else {
      changeData = new Float32Array(currentFrame.length)
    }

    this.lastFrame = currentFrame

    return changeData
  }

  private preprocessChangeImage(imageData: ImageData): Float32Array {
    const { width, height, data } = imageData
    const size = CONFIG.inputSize
    const result = new Float32Array(size * size * 3)
    const channelStride = size * size

    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        const srcIdx = (h * width + w) * 4
        const pixelIdx = h * width + w

        result[pixelIdx] = data[srcIdx + 0] / 255.0
        result[channelStride + pixelIdx] = data[srcIdx + 1] / 255.0
        result[channelStride * 2 + pixelIdx] = data[srcIdx + 2] / 255.0
      }
    }
    return result
  }

  private computeFrameDifference(prevFrame: Float32Array, currFrame: Float32Array): Float32Array {
    const size = CONFIG.inputSize * CONFIG.inputSize * 3
    const diff = new Float32Array(size)

    for (let i = 0; i < size; i++) {
      diff[i] = currFrame[i] - prevFrame[i]
    }

    return diff
  }

  private buildInputTensors(): { pose_seq: Tensor; change_seq: Tensor } {
    const numFrames = CONFIG.numFrames
    const size = CONFIG.inputSize
    const channels = 3

    const poseFrames = this.poseBuffer.toArray()
    const changeFrames = this.changeBuffer.toArray()

    this.poseSeqData.fill(0)
    this.changeSeqData.fill(0)

    const actualFrames = Math.min(poseFrames.length, numFrames)
    for (let f = 0; f < actualFrames; f++) {
      const frame = poseFrames[f]
      for (let i = 0; i < frame.length; i++) {
        this.poseSeqData[f * channels * size * size + i] = frame[i]
      }
    }

    const actualChangeFrames = Math.min(changeFrames.length, numFrames)
    for (let f = 0; f < actualChangeFrames; f++) {
      const diff = changeFrames[f]
      for (let i = 0; i < diff.length; i++) {
        this.changeSeqData[f * channels * size * size + i] = diff[i]
      }
    }

    const pose_seq = new Tensor('float32', this.poseSeqData, [1, numFrames, channels, size, size])
    const change_seq = new Tensor('float32', this.changeSeqData, [1, numFrames, channels, size, size])

    return { pose_seq, change_seq }
  }

  private async runInference(): Promise<void> {
    const now = Date.now()

    if (now - this.lastInferenceTime < 500) {
      return
    }

    if (!this.violenceSession) {
      console.warn('[Worker] No violence session')
      return
    }

    if (this.poseBuffer.getLength() < CONFIG.minFrames) {
      console.log('[Worker] Not enough frames:', this.poseBuffer.getLength())
      return
    }

    try {
      console.log('[Worker] Running inference with', this.poseBuffer.getLength(), 'frames')

      const { pose_seq, change_seq } = this.buildInputTensors()

      console.log('[Worker] Input shapes - pose:', pose_seq.dims, 'change:', change_seq.dims)

      const feeds: Record<string, Tensor> = {
        'pose_seq': pose_seq,
        'change_seq': change_seq
      }

      const results = await this.violenceSession.run(feeds)

      console.log('[Worker] Inference results keys:', Object.keys(results))

      const outputTensor = results['prediction'] as Tensor

      const score = outputTensor.data[0] as number

      pose_seq.dispose()
      change_seq.dispose()

      this.lastInferenceTime = now

      console.log('[Worker] Inference score:', score)

      const result: InferenceResult = {
        score,
        isFight: score > CONFIG.threshold,
        timestamp: now
      }

      this.postMessage({
        type: 'result',
        data: result
      })
    } catch (error) {
      console.error('[Worker] Inference error:', error)
      this.postMessage({ type: 'error', data: (error as Error).message })
    }
  }

  reset(): void {
    this.poseBuffer.clear()
    this.changeBuffer.clear()
    this.lastFrame = null
    this.frameCounter = 0
    this.lastInferenceTime = 0
  }

  updateConfig(newConfig: Partial<typeof CONFIG>): void {
    Object.assign(CONFIG, newConfig)
  }

  getStatus(): { poseBufferLength: number; changeBufferLength: number; isProcessing: boolean } {
    return {
      poseBufferLength: this.poseBuffer.getLength(),
      changeBufferLength: this.changeBuffer.getLength(),
      isProcessing: this.isProcessing
    }
  }
}

const worker = new ViolenceDetectionWorker()

worker.initialize()
  .then(() => {
    console.log('[Worker] Initialization complete')
  })
  .catch((error) => {
    console.error('[Worker] Initialization failed:', error)
  })

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const { type, data } = event.data

  switch (type) {
    case 'frame': {
      try {
        let imageBitmap: ImageBitmap

        if (data instanceof ImageBitmap) {
          imageBitmap = data
        } else if (data.frame instanceof Uint8ClampedArray) {
          const width = data.width || 640
          const height = data.height || 480
          const imageData = new ImageData(data.frame, width, height)
          imageBitmap = await createImageBitmap(imageData)
        } else if (data.frame instanceof ArrayBuffer) {
          const blob = new Blob([data.frame], { type: 'image/png' })
          imageBitmap = await createImageBitmap(blob)
        } else {
          console.error('[Worker] Unknown frame data type')
          return
        }

        await worker.processFrame(imageBitmap)
        imageBitmap.close()
      } catch (error) {
        console.error('[Worker] Failed to process frame:', error)
      }
      break
    }

    case 'reset': {
      worker.reset()
      break
    }

    case 'config': {
      worker.updateConfig(data)
      break
    }

    case 'init': {
      break
    }

    default:
      console.warn('[Worker] Unknown message type:', type)
  }
}

export { }
