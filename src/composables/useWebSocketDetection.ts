import { ref, onUnmounted } from 'vue'

export interface WebSocketDetectionResult {
    score: number
    isFight: boolean
    inferenceTimeMs: number
    timestamp: number
}

export interface UseWebSocketDetectionOptions {
    url?: string
    inferenceFps?: number
    frameQuality?: number
    frameWidth?: number
    frameHeight?: number
    onDetection?: (result: WebSocketDetectionResult) => void
    onConnect?: () => void
    onDisconnect?: () => void
    onError?: (error: Event) => void
}

export function useWebSocketDetection(options: UseWebSocketDetectionOptions = {}) {
    const url = options.url || 'ws://localhost:8000/ws/detect'
    const inferenceFps = options.inferenceFps || 20
    const frameQuality = options.frameQuality || 0.8
    const frameWidth = options.frameWidth || 640
    const frameHeight = options.frameHeight || 640

    const ws = ref<WebSocket | null>(null)
    const isConnected = ref(false)
    const isProcessing = ref(false)
    const lastResult = ref<WebSocketDetectionResult | null>(null)
    const error = ref<string | null>(null)
    const bufferProgress = ref(0)

    let animationFrameId: number | null = null
    let canvas: HTMLCanvasElement | null = null
    let ctx: CanvasRenderingContext2D | null = null

    const connect = (): Promise<void> => {
        return new Promise((resolve, reject) => {
            try {
                ws.value = new WebSocket(url)

                ws.value.onopen = () => {
                    console.log('[WebSocket] Connected to server')
                    isConnected.value = true
                    options.onConnect?.()
                    resolve()
                }

                ws.value.onclose = () => {
                    console.log('[WebSocket] Disconnected from server')
                    isConnected.value = false
                    options.onDisconnect?.()
                }

                ws.value.onerror = (event) => {
                    console.error('[WebSocket] Error:', event)
                    error.value = 'WebSocket connection error'
                    options.onError?.(event)
                    reject(event)
                }

                ws.value.onmessage = (event) => {
                    try {
                        const message = JSON.parse(event.data)

                        if (message.type === 'result') {
                            const result = message.data as WebSocketDetectionResult
                            lastResult.value = result
                            bufferProgress.value = 1.0
                            options.onDetection?.(result)
                        } else if (message.type === 'buffer') {
                            bufferProgress.value = message.data.progress
                        }
                    } catch (e) {
                        console.error('[WebSocket] Failed to parse message:', e)
                    }
                }
            } catch (e) {
                reject(e)
            }
        })
    }

    const disconnect = (): void => {
        console.log('[WS] disconnect called')
        if (animationFrameId !== null) {
            cancelAnimationFrame(animationFrameId)
            animationFrameId = null
        }

        if (ws.value) {
            ws.value.close()
            ws.value = null
        }

        isConnected.value = false
    }

    const sendFrame = (videoElement: HTMLVideoElement): void => {
        console.log('[WS] sendFrame called, ws=', !!ws.value, 'connected=', isConnected.value)
        if (!ws.value || !isConnected.value) {
            console.log('[WS] Skipping frame: ws=', !!ws.value, 'connected=', isConnected.value)
            return
        }

        isProcessing.value = true

        if (!canvas) {
            canvas = document.createElement('canvas')
            canvas.width = frameWidth
            canvas.height = frameHeight
            ctx = canvas.getContext('2d')
        }

        if (!ctx) {
            return
        }

        const videoWidth = videoElement.videoWidth
        const videoHeight = videoElement.videoHeight

        if (videoWidth === 0 || videoHeight === 0) {
            isProcessing.value = false
            return
        }

        const scale = Math.max(frameWidth / videoWidth, frameHeight / videoHeight)
        const scaledWidth = videoWidth * scale
        const scaledHeight = videoHeight * scale
        const offsetX = (frameWidth - scaledWidth) / 2
        const offsetY = (frameHeight - scaledHeight) / 2

        ctx.fillStyle = '#000'
        ctx.fillRect(0, 0, frameWidth, frameHeight)
        ctx.drawImage(videoElement, offsetX, offsetY, scaledWidth, scaledHeight)

        canvas.toBlob(
            (blob) => {
                if (!blob || !ws.value || ws.value.readyState !== WebSocket.OPEN) {
                    isProcessing.value = false
                    return
                }

                const reader = new FileReader()
                reader.onloadend = () => {
                    const base64 = (reader.result as string).split(',')[1]

                    ws.value?.send(
                        JSON.stringify({
                            type: 'frame',
                            data: {
                                image: base64,
                                width: videoWidth,
                                height: videoHeight,
                                timestamp: Date.now()
                            }
                        })
                    )

                    isProcessing.value = false
                }
                reader.readAsDataURL(blob)
            },
            'image/jpeg',
            frameQuality
        )
    }

    const reset = (): void => {
        if (ws.value && isConnected.value) {
            ws.value.send(JSON.stringify({ type: 'reset' }))
        }
        lastResult.value = null
        bufferProgress.value = 0
    }

    const startContinuousDetection = (
        videoElement: HTMLVideoElement,
        videoCallback?: () => HTMLVideoElement | null
    ): void => {
        console.log('[WS] startContinuousDetection called')
        const frameInterval = 1000 / inferenceFps
        let lastFrameTime = 0

        const detect = (timestamp: number): void => {
            let video = videoElement

            if (videoCallback) {
                const cbVideo = videoCallback()
                if (cbVideo) {
                    video = cbVideo
                }
            }

            if (
                video &&
                !video.paused &&
                !video.ended &&
                video.readyState >= 2 &&
                timestamp - lastFrameTime >= frameInterval
            ) {
                console.log('[WS] Sending frame, paused=', video.paused, 'ended=', video.ended, 'readyState=', video.readyState)
                sendFrame(video)
                lastFrameTime = timestamp
            }

            animationFrameId = requestAnimationFrame(detect)
        }

        animationFrameId = requestAnimationFrame(detect)
    }

    const stopContinuousDetection = (): void => {
        console.log('[WS] stopContinuousDetection called')
        if (animationFrameId !== null) {
            cancelAnimationFrame(animationFrameId)
            animationFrameId = null
        }
    }

    const sendPing = (): void => {
        if (ws.value && isConnected.value) {
            ws.value.send(JSON.stringify({ type: 'ping' }))
        }
    }

    onUnmounted(() => {
        disconnect()
    })

    return {
        isConnected,
        isProcessing,
        lastResult,
        error,
        bufferProgress,
        connect,
        disconnect,
        sendFrame,
        reset,
        startContinuousDetection,
        stopContinuousDetection,
        sendPing
    }
}
