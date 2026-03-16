let ffmpeg: any = null
let isLoading = false

export interface TranscodeProgress {
  phase: 'loading' | 'transcoding' | 'complete' | 'error'
  progress: number
  message: string
}

async function loadFFmpegCore(): Promise<any> {
  if (typeof window === 'undefined') return null
  
  const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm'
  
  const script1 = document.createElement('script')
  script1.src = `${baseURL}/ffmpeg-core.js`
  script1.crossOrigin = 'anonymous'
  
  await new Promise<void>((resolve, reject) => {
    script1.onload = () => resolve()
    script1.onerror = () => reject(new Error('Failed to load ffmpeg-core.js'))
    document.head.appendChild(script1)
  })

  return (window as any).FFmpeg
}

export async function getFFmpeg(
  onProgress?: (progress: TranscodeProgress) => void
): Promise<any> {
  if (ffmpeg) return ffmpeg

  if (isLoading) {
    return new Promise((resolve) => {
      const checkInterval = setInterval(() => {
        if (ffmpeg) {
          clearInterval(checkInterval)
          resolve(ffmpeg)
        }
      }, 500)
    })
  }

  isLoading = true

  try {
    onProgress?.({
      phase: 'loading',
      progress: 0,
      message: 'Loading FFmpeg core...'
    })

    const FFmpeg = await loadFFmpegCore()
    
    if (!FFmpeg) {
      throw new Error('Failed to load FFmpeg')
    }

    ffmpeg = new FFmpeg()

    ffmpeg.on('log', ({ message }: { message: string }) => {
      console.log('[FFmpeg]', message)
    })

    ffmpeg.on('progress', ({ progress }: { progress: number }) => {
      onProgress?.({
        phase: 'transcoding',
        progress: Math.round(progress * 100),
        message: `Transcoding: ${Math.round(progress * 100)}%`
      })
    })

    await ffmpeg.load()

    onProgress?.({
      phase: 'loading',
      progress: 100,
      message: 'FFmpeg loaded successfully'
    })

    return ffmpeg
  } catch (error) {
    console.error('Failed to load FFmpeg:', error)
    onProgress?.({
      phase: 'error',
      progress: 0,
      message: `Failed to load FFmpeg: ${(error as Error).message}`
    })
    return null
  } finally {
    isLoading = false
  }
}

export async function transcodeVideo(
  file: File,
  onProgress?: (progress: TranscodeProgress) => void
): Promise<{ url: string; name: string } | null> {
  const ff = await getFFmpeg(onProgress)
  if (!ff) return null

  try {
    onProgress?.({
      phase: 'transcoding',
      progress: 0,
      message: 'Starting transcoding...'
    })

    const inputName = `input_${Date.now()}.avi`
    const outputName = `output_${Date.now()}.mp4`

    const fileData = await file.arrayBuffer()
    await ff.writeFile(inputName, new Uint8Array(fileData))

    await ff.exec([
      '-i', inputName,
      '-c:v', 'libx264',
      '-preset', 'fast',
      '-crf', '23',
      '-c:a', 'aac',
      '-b:a', '128k',
      '-movflags', '+faststart',
      '-y',
      outputName
    ])

    const data = await ff.readFile(outputName)
    const blob = new Blob([data], { type: 'video/mp4' })
    const url = URL.createObjectURL(blob)

    await ff.deleteFile(inputName)
    await ff.deleteFile(outputName)

    onProgress?.({
      phase: 'complete',
      progress: 100,
      message: 'Transcoding complete'
    })

    return {
      url,
      name: file.name.replace(/\.avi$/i, '') + '.mp4'
    }
  } catch (error) {
    console.error('Transcoding error:', error)
    onProgress?.({
      phase: 'error',
      progress: 0,
      message: `Transcoding failed: ${(error as Error).message}`
    })
    return null
  }
}

export function revokeVideoUrl(url: string): void {
  if (url.startsWith('blob:')) {
    URL.revokeObjectURL(url)
  }
}
