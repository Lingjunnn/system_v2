import { ref, shallowRef, onUnmounted, computed } from 'vue'

export interface InferenceResult {
  score: number
  isFight: boolean
  timestamp: number
}

export interface WorkerDetectionState {
  isModelLoaded: boolean
  isProcessing: boolean
  bufferProgress: number
  lastResult: InferenceResult | null
  fps: number
  error: string | null
  yoloLoaded: boolean
}

export interface UseWorkerDetectionOptions {
  numFrames?: number
  inputSize?: number
  stride?: number
  threshold?: number
  onDetection?: (result: InferenceResult) => void
}

export function useWorkerDetection(options: UseWorkerDetectionOptions = {}) {
  const worker = shallowRef<Worker | null>(null)

  const state = ref<WorkerDetectionState>({
    isModelLoaded: false,
    isProcessing: false,
    bufferProgress: 0,
    lastResult: null,
    fps: 0,
    error: null,
    yoloLoaded: false
  })

  const frameCount = ref(0)
  const lastFpsUpdate = ref(Date.now())
  const animationFrameId = ref<number | null>(null)

  const isReady = computed(() => state.value.isModelLoaded)

  const initialize = async (): Promise<boolean> => {
    if (worker.value && state.value.isModelLoaded) {
      console.log('[Worker] Already initialized, reusing existing worker')
      return true
    }

    if (worker.value) {
      console.log('[Worker] Terminating existing worker before reinitialize')
      worker.value.terminate()
      worker.value = null
    }

    try {
      state.value.error = null
      state.value.isModelLoaded = false

      const workerPath = new URL('../workers/inference.worker.ts', import.meta.url)
      worker.value = new Worker(workerPath, { type: 'module' })

      worker.value.onmessage = handleWorkerMessage
      worker.value.onerror = handleWorkerError

      return new Promise((resolve) => {
        const timeout = setTimeout(() => {
          console.warn('[Worker] Initialization timeout - checking worker status...')
          if (worker.value) {
            worker.value.terminate()
          }
          resolve(false)
        }, 60000)

        if (!worker.value) {
          resolve(false)
          return
        }

        worker.value.onmessage = (e: MessageEvent) => {
          const msg = e.data
          console.log('[Main] Worker message:', msg?.type, msg?.data)

          if (msg?.type === 'ready') {
            clearTimeout(timeout)
            state.value.isModelLoaded = true
            state.value.yoloLoaded = msg.data?.yoloLoaded ?? false
            console.log('[Main] Worker ready, YOLO loaded:', state.value.yoloLoaded)
            resolve(true)
          } else if (msg?.type === 'error') {
            clearTimeout(timeout)
            state.value.error = msg.data
            console.error('[Main] Worker error:', msg.data)
            resolve(false)
          } else {
            handleWorkerMessage(e)
          }
        }
      })
    } catch (error) {
      state.value.error = `Failed to initialize worker: ${(error as Error).message}`
      console.error('[Main] Worker initialization failed:', error)
      return false
    }
  }

  const handleWorkerMessage = (event: MessageEvent) => {
    const { type, data } = event.data

    switch (type) {
      case 'ready': {
        state.value.isModelLoaded = true
        state.value.yoloLoaded = data?.yoloLoaded ?? false
        console.log('[Main] Worker ready received')
        break
      }

      case 'result': {
        state.value.isProcessing = false
        state.value.lastResult = data

        if (options.onDetection) {
          options.onDetection(data)
        }
        break
      }

      case 'buffer': {
        state.value.bufferProgress = data.progress
        break
      }

      case 'error': {
        state.value.error = data
        state.value.isProcessing = false
        console.error('[Main] Worker error:', data)
        break
      }

      case 'status': {
        state.value.isProcessing = data.isProcessing
        break
      }

      default:
        break
    }
  }

  const handleWorkerError = (error: Event) => {
    console.error('[Main] Worker error event:', error)
    state.value.error = 'Worker crashed'
    state.value.isProcessing = false
  }

  const sendFrame = async (videoElement: HTMLVideoElement): Promise<void> => {
    if (!worker.value || !state.value.isModelLoaded) {
      return
    }

    try {
      const canvas = new OffscreenCanvas(videoElement.videoWidth, videoElement.videoHeight)
      const ctx = canvas.getContext('2d')

      if (!ctx) {
        console.error('[Main] Failed to get offscreen canvas context')
        return
      }

      ctx.drawImage(videoElement, 0, 0)

      const frame = await canvas.convertToBlob({
        type: 'image/png',
        quality: 1.0
      })

      const arrayBuffer = await frame.arrayBuffer()

      worker.value.postMessage({
        type: 'frame',
        data: {
          frame: arrayBuffer,
          width: videoElement.videoWidth,
          height: videoElement.videoHeight
        }
      }, [arrayBuffer])

      frameCount.value++
      updateFps()
    } catch (error) {
      console.error('[Main] Failed to send frame:', error)
    }
  }

  const sendFrameTransfer = async (videoElement: HTMLVideoElement): Promise<void> => {
    if (!worker.value || !state.value.isModelLoaded) {
      return
    }

    try {
      const width = videoElement.videoWidth || 640
      const height = videoElement.videoHeight || 480

      const canvas = new OffscreenCanvas(width, height)
      const ctx = canvas.getContext('2d')

      if (!ctx) {
        console.error('[Main] Failed to get offscreen canvas context')
        return
      }

      ctx.drawImage(videoElement, 0, 0)

      const imageData = ctx.getImageData(0, 0, width, height)

      const buffer = new Uint8ClampedArray(imageData.data.buffer)
      const transferableBuffer = buffer.buffer

      worker.value.postMessage({
        type: 'frame',
        data: {
          frame: transferableBuffer,
          width,
          height
        }
      }, [transferableBuffer])

      frameCount.value++
      updateFps()
    } catch (error) {
      console.error('[Main] Failed to send frame (transfer):', error)
    }
  }

  const sendFrameImageBitmap = async (videoElement: HTMLVideoElement): Promise<void> => {
    if (!worker.value || !state.value.isModelLoaded) {
      return
    }

    if (state.value.isProcessing) {
      return
    }

    if (!videoElement || videoElement.readyState < 2 || videoElement.paused || videoElement.ended) {
      return
    }

    try {
      const imageBitmap = await createImageBitmap(videoElement)

      worker.value.postMessage({
        type: 'frame',
        data: imageBitmap
      }, [imageBitmap])

      frameCount.value++
      updateFps()
    } catch (error) {
      // ? createImageBitmap 
    }
  }

  const reset = (): void => {
    if (worker.value) {
      worker.value.postMessage({ type: 'reset' })
    }
    state.value.bufferProgress = 0
    state.value.lastResult = null
    frameCount.value = 0
  }

  const updateConfig = (config: Partial<UseWorkerDetectionOptions>): void => {
    if (worker.value) {
      worker.value.postMessage({
        type: 'config',
        data: config
      })
    }
  }

  const updateFps = (): void => {
    const now = Date.now()
    const elapsed = now - lastFpsUpdate.value

    if (elapsed >= 1000) {
      const fps = Math.round((frameCount.value * 1000) / elapsed)
      state.value.fps = fps
      frameCount.value = 0
      lastFpsUpdate.value = now
    }
  }

  const startContinuousDetection = (
    videoElement: HTMLVideoElement,
    videoCallback?: () => HTMLVideoElement | null
  ): void => {
    const detect = async () => {
      let video = videoElement

      if (videoCallback) {
        video = videoCallback() || videoElement
      }

      if (video && video.paused === false && video.ended === false) {
        await sendFrameImageBitmap(video)
      }

      animationFrameId.value = requestAnimationFrame(detect)
    }

    detect()
  }

  const stopContinuousDetection = (): void => {
    if (animationFrameId.value !== null) {
      cancelAnimationFrame(animationFrameId.value)
      animationFrameId.value = null
    }
  }

  const dispose = async (): Promise<void> => {
    stopContinuousDetection()

    if (worker.value) {
      worker.value.terminate()
      worker.value = null
    }

    state.value = {
      isModelLoaded: false,
      isProcessing: false,
      bufferProgress: 0,
      lastResult: null,
      fps: 0,
      error: null,
      yoloLoaded: false
    }
  }

  onUnmounted(() => {
    dispose()
  })

  return {
    state,
    isReady,
    initialize,
    sendFrame,
    sendFrameTransfer,
    sendFrameImageBitmap,
    reset,
    updateConfig,
    startContinuousDetection,
    stopContinuousDetection,
    dispose
  }
}
