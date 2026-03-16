import { ref, shallowRef, onUnmounted } from 'vue'
import { ViolenceDetector, InferenceResult, ModelConfig } from '@/utils/inference/onnx'

export interface UseViolenceDetectionOptions {
    modelPath?: string
    numFrames?: number
    inputSize?: number
    stride?: number
    threshold?: number
    inferenceFps?: number
    onDetection?: (result: InferenceResult) => void
}

export interface DetectionState {
    isModelLoaded: boolean
    isProcessing: boolean
    bufferProgress: number
    lastResult: InferenceResult | null
    fps: number
    error: string | null
}

export function useViolenceDetection(options: UseViolenceDetectionOptions = {}) {
    const detector = shallowRef<ViolenceDetector | null>(null)

    const state = ref<DetectionState>({
        isModelLoaded: false,
        isProcessing: false,
        bufferProgress: 0,
        lastResult: null,
        fps: 0,
        error: null
    })

    const frameCount = ref(0)
    const lastFpsUpdate = ref(Date.now())
    const animationFrameId = ref<number | null>(null)

    const initialize = async (): Promise<boolean> => {
        try {
            state.value.error = null
            state.value.isModelLoaded = false

            const config: Partial<ModelConfig> = {
                modelPath: options.modelPath || '/models/violence_detector_standalone.onnx',
                numFrames: options.numFrames || 50,
                inputSize: options.inputSize || 100,
                stride: options.stride || 10,
                threshold: options.threshold || 0.5
            }

            detector.value = new ViolenceDetector(config)
            await detector.value.initialize()

            state.value.isModelLoaded = true
            console.log('Violence detection model initialized successfully')
            return true
        } catch (error) {
            state.value.error = `Failed to load model: ${(error as Error).message}`
            console.error('Failed to initialize violence detector:', error)
            return false
        }
    }

    const addFrame = (videoElement: HTMLVideoElement): boolean => {
        if (!detector.value || !state.value.isModelLoaded) {
            return false
        }

        try {
            const isReady = detector.value.addFrame(videoElement)
            state.value.bufferProgress = detector.value.getBufferProgress()
            frameCount.value++

            updateFps()
            return isReady
        } catch (error) {
            console.error('Error adding frame:', error)
            return false
        }
    }

    const runInference = async (): Promise<InferenceResult | null> => {
        if (!detector.value || !state.value.isModelLoaded) {
            return null
        }

        if (state.value.isProcessing) {
            return null
        }

        state.value.isProcessing = true

        try {
            const result = await detector.value.infer()

            if (result) {
                state.value.lastResult = result

                if (options.onDetection) {
                    options.onDetection(result)
                }
            }

            return result
        } catch (error) {
            console.error('Inference error:', error)
            state.value.error = `Inference error: ${(error as Error).message}`
            return null
        } finally {
            state.value.isProcessing = false
        }
    }

    const processFrame = async (videoElement: HTMLVideoElement): Promise<InferenceResult | null> => {
        const isReady = addFrame(videoElement)

        if (isReady) {
            return await runInference()
        }

        return null
    }

    const startContinuousDetection = (
        _videoElement: HTMLVideoElement,
        videoCallback: () => HTMLVideoElement | null
    ) => {
        const detect = async () => {
            const video = videoCallback()
            if (video && video.paused === false && video.ended === false) {
                await processFrame(video)
            }

            animationFrameId.value = requestAnimationFrame(detect)
        }

        detect()
    }

    const stopContinuousDetection = () => {
        if (animationFrameId.value !== null) {
            cancelAnimationFrame(animationFrameId.value)
            animationFrameId.value = null
        }
    }

    const updateFps = () => {
        const now = Date.now()
        const elapsed = now - lastFpsUpdate.value

        if (elapsed >= 1000) {
            state.value.fps = Math.round((frameCount.value * 1000) / elapsed)
            frameCount.value = 0
            lastFpsUpdate.value = now
        }
    }

    const reset = () => {
        if (detector.value) {
            detector.value.reset()
            state.value.bufferProgress = 0
            state.value.lastResult = null
            frameCount.value = 0
        }
    }

    const dispose = async () => {
        stopContinuousDetection()

        if (detector.value) {
            await detector.value.dispose()
            detector.value = null
        }

        state.value = {
            isModelLoaded: false,
            isProcessing: false,
            bufferProgress: 0,
            lastResult: null,
            fps: 0,
            error: null
        }
    }

    onUnmounted(() => {
        dispose()
    })

    return {
        state,
        initialize,
        addFrame,
        runInference,
        processFrame,
        startContinuousDetection,
        stopContinuousDetection,
        reset,
        dispose
    }
}
