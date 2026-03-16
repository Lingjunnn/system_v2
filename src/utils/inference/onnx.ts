import { InferenceSession, Tensor, env } from 'onnxruntime-web'

  ; (env as any).wasm = {
    numThreads: 1,
    simd: false,
    wasmPaths: '/node_modules/onnxruntime-web/dist/'
  }

export interface ModelConfig {
  modelPath: string
  numFrames: number
  inputSize: number
  stride: number
  threshold: number
}

export interface InferenceResult {
  score: number
  isFight: boolean
  timestamp: number
}

const DEFAULT_CONFIG: ModelConfig = {
  modelPath: '../models/violence_detector_standalone.onnx',
  numFrames: 50,
  inputSize: 100,
  stride: 10,
  threshold: 0.5
}

let ortModule: typeof import('onnxruntime-web') | null = null

async function getOrtModule(): Promise<typeof import('onnxruntime-web')> {
  if (!ortModule) {
    ortModule = await import('onnxruntime-web')
  }
  return ortModule
}

export class ViolenceDetector {
  private session: InferenceSession | null = null
  private config: ModelConfig
  private frameBuffer: Float32Array[]
  private isProcessing = false
  private lastInferenceTime = 0
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D

  constructor(config: Partial<ModelConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config }
    this.frameBuffer = []

    this.canvas = document.createElement('canvas')
    this.canvas.width = this.config.inputSize
    this.canvas.height = this.config.inputSize
    this.ctx = this.canvas.getContext('2d')!
  }

  async initialize(): Promise<void> {
    try {
      const ortModule = await getOrtModule()
      console.log('[ONNX] Initializing ONNX Runtime Web...')
      console.log('[ONNX] Loading model from:', this.config.modelPath)

      this.session = await ortModule.InferenceSession.create(this.config.modelPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      })

      console.log('[ONNX] Violence detection model loaded successfully')
    } catch (error) {
      console.error('[ONNX] Failed to load ONNX model:', error)
      console.error('[ONNX] Error details:', JSON.stringify(error))
      throw error
    }
  }

  addFrame(videoElement: HTMLVideoElement): boolean {
    if (this.isProcessing) return false

    this.ctx.drawImage(
      videoElement,
      0, 0,
      this.config.inputSize,
      this.config.inputSize
    )

    const imageData = this.ctx.getImageData(0, 0, this.config.inputSize, this.config.inputSize)
    const preprocessed = this.preprocessFrame(imageData)

    this.frameBuffer.push(preprocessed)

    if (this.frameBuffer.length > this.config.numFrames) {
      this.frameBuffer.shift()
    }

    return this.frameBuffer.length >= this.config.numFrames
  }

  private preprocessFrame(imageData: ImageData): Float32Array {
    const { width, height, data } = imageData
    const size = this.config.inputSize

    const result = new Float32Array(size * size * 3)

    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        const srcIdx = (h * width + w) * 4
        const dstIdxRGB = (h * width + w) * 3

        result[dstIdxRGB + 0] = data[srcIdx + 0] / 255.0
        result[dstIdxRGB + 1] = data[srcIdx + 1] / 255.0
        result[dstIdxRGB + 2] = data[srcIdx + 2] / 255.0
      }
    }

    return result
  }

  private computeFrameDifference(prevFrame: Float32Array, currFrame: Float32Array): Float32Array {
    const size = this.config.inputSize * this.config.inputSize * 3
    const diff = new Float32Array(size)

    for (let i = 0; i < size; i++) {
      diff[i] = Math.abs(currFrame[i] - prevFrame[i])
    }

    return diff
  }

  private buildInputTensors(): {
    pose_seq: Tensor
    change_seq: Tensor
  } {
    const numFrames = this.config.numFrames
    const size = this.config.inputSize

    const poseSeqData = new Float32Array(1 * numFrames * 3 * size * size)
    const changeSeqData = new Float32Array(1 * numFrames * 3 * size * size)

    for (let f = 0; f < numFrames; f++) {
      const frame = this.frameBuffer[f]
      for (let i = 0; i < size * size * 3; i++) {
        poseSeqData[f * size * size * 3 + i] = frame[i]
      }
    }

    for (let f = 1; f < numFrames; f++) {
      const prevFrame = this.frameBuffer[f - 1]
      const currFrame = this.frameBuffer[f]
      const diff = this.computeFrameDifference(prevFrame, currFrame)
      for (let i = 0; i < size * size * 3; i++) {
        changeSeqData[f * size * size * 3 + i] = diff[i]
      }
    }

    changeSeqData.fill(0, 0, size * size * 3)

    const pose_seq = new Tensor('float32', poseSeqData, [1, numFrames, 3, size, size])
    const change_seq = new Tensor('float32', changeSeqData, [1, numFrames, 3, size, size])

    return { pose_seq, change_seq }
  }

  async infer(): Promise<InferenceResult | null> {
    if (!this.session || this.frameBuffer.length < this.config.numFrames) {
      return null
    }

    const now = Date.now()
    if (now - this.lastInferenceTime < 1000 / 5) {
      return null
    }

    this.isProcessing = true

    try {
      const { pose_seq, change_seq } = this.buildInputTensors()

      const feeds: Record<string, Tensor> = {
        'pose_seq': pose_seq,
        'change_seq': change_seq
      }

      const results = await this.session.run(feeds)
      const outputTensor = results['prediction'] as Tensor
      const score = outputTensor.data[0] as number

      pose_seq.dispose()
      change_seq.dispose()

      this.lastInferenceTime = now

      return {
        score,
        isFight: score > this.config.threshold,
        timestamp: now
      }
    } catch (error) {
      console.error('Inference error:', error)
      return null
    } finally {
      this.isProcessing = false
    }
  }

  reset(): void {
    this.frameBuffer = []
    this.lastInferenceTime = 0
  }

  getBufferLength(): number {
    return this.frameBuffer.length
  }

  getBufferProgress(): number {
    return this.frameBuffer.length / this.config.numFrames
  }

  isReady(): boolean {
    return this.session !== null && this.frameBuffer.length >= this.config.numFrames
  }

  async dispose(): Promise<void> {
    if (this.session) {
      this.session.release()
      this.session = null
    }
    this.frameBuffer = []
  }
}

export async function createViolenceDetector(config?: Partial<ModelConfig>): Promise<ViolenceDetector> {
  const detector = new ViolenceDetector(config)
  await detector.initialize()
  return detector
}

export function drawDetectionResult(
  ctx: CanvasRenderingContext2D,
  result: InferenceResult,
  width: number,
  height: number
): void {
  ctx.clearRect(0, 0, width, height)

  if (!result) return

  const color = result.isFight ? '#ff4444' : '#00ff88'
  const label = result.isFight ? 'FIGHT DETECTED' : 'NORMAL'
  const confidence = (result.score * 100).toFixed(1)

  ctx.strokeStyle = color
  ctx.lineWidth = 3
  ctx.strokeRect(10, 10, width - 20, height - 20)

  ctx.fillStyle = color
  ctx.font = 'bold 24px Arial'
  ctx.fillText(`${label} (${confidence}%)`, 20, 50)

  if (result.isFight) {
    ctx.fillStyle = 'rgba(255, 0, 0, 0.2)'
    ctx.fillRect(0, 0, width, height)
  }
}
