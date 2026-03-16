<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted, computed } from 'vue'

interface VideoInfo {
  name: string
  duration: number
  width: number
  height: number
  frameRate: number
}

const props = defineProps<{
  src?: string
  autoplay?: boolean
  muted?: boolean
}>()

const emit = defineEmits<{
  (e: 'play'): void
  (e: 'pause'): void
  (e: 'ended'): void
  (e: 'timeupdate', currentTime: number): void
  (e: 'loaded', info: VideoInfo): void
  (e: 'error', error: Error): void
}>()

const videoRef = ref<HTMLVideoElement | null>(null)
const containerRef = ref<HTMLDivElement | null>(null)
const isPlaying = ref(false)
const isLoading = ref(false)
const currentTime = ref(0)
const duration = ref(0)
const volume = ref(1)
const isMuted = ref(props.muted ?? false)
const isFullscreen = ref(false)
const videoError = ref<string | null>(null)
const playbackRate = ref(1)

const videoInfo = ref<VideoInfo | null>(null)

const formattedCurrentTime = computed(() => formatTime(currentTime.value))
const formattedDuration = computed(() => formatTime(duration.value))

const progress = computed(() => {
  if (duration.value === 0) return 0
  return (currentTime.value / duration.value) * 100
})

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

const loadVideo = (src: string) => {
  if (!videoRef.value) return

  isLoading.value = true
  videoError.value = null
  videoRef.value.src = src
  videoRef.value.load()
}

const play = () => {
  if (!videoRef.value) return
  videoRef.value.play()
}

const pause = () => {
  if (!videoRef.value) return
  videoRef.value.pause()
}

const togglePlay = () => {
  if (isPlaying.value) {
    pause()
  } else {
    play()
  }
}

const seek = (event: MouseEvent) => {
  if (!videoRef.value || duration.value === 0) return

  const progressBar = event.currentTarget as HTMLElement
  const rect = progressBar.getBoundingClientRect()
  const pos = (event.clientX - rect.left) / rect.width
  videoRef.value.currentTime = pos * duration.value
}

const changeVolume = (event: Event) => {
  if (!videoRef.value) return

  const target = event.target as HTMLInputElement
  const value = parseFloat(target.value)
  volume.value = value
  videoRef.value.volume = value
  isMuted.value = value === 0
}

const toggleMute = () => {
  if (!videoRef.value) return

  isMuted.value = !isMuted.value
  videoRef.value.muted = isMuted.value
}

const toggleFullscreen = async () => {
  if (!containerRef.value) return

  try {
    if (!document.fullscreenElement) {
      await containerRef.value.requestFullscreen()
      isFullscreen.value = true
    } else {
      await document.exitFullscreen()
      isFullscreen.value = false
    }
  } catch (err) {
    console.error('Fullscreen error:', err)
  }
}

const setPlaybackRate = (rate: number) => {
  if (!videoRef.value) return
  playbackRate.value = rate
  videoRef.value.playbackRate = rate
}

const skip = (seconds: number) => {
  if (!videoRef.value) return
  videoRef.value.currentTime = Math.max(0, Math.min(duration.value, currentTime.value + seconds))
}

const handlePlay = () => {
  isPlaying.value = true
  emit('play')
}

const handlePause = () => {
  isPlaying.value = false
  emit('pause')
}

const handleEnded = () => {
  isPlaying.value = false
  emit('ended')
}

const handleTimeUpdate = () => {
  if (!videoRef.value) return
  currentTime.value = videoRef.value.currentTime
  emit('timeupdate', currentTime.value)
}

const handleLoadedMetadata = () => {
  if (!videoRef.value) return

  duration.value = videoRef.value.duration

  const info: VideoInfo = {
    name: '',
    duration: videoRef.value.duration,
    width: videoRef.value.videoWidth,
    height: videoRef.value.videoHeight,
    frameRate: 30
  }

  videoInfo.value = info
  isLoading.value = false
  emit('loaded', info)
}

const handleError = () => {
  let errorMsg = 'Failed to load video'

  if (videoRef.value?.error) {
    const error = videoRef.value.error
    switch (error.code) {
      case MediaError.MEDIA_ERR_ABORTED:
        errorMsg = 'Video playback was aborted'
        break
      case MediaError.MEDIA_ERR_NETWORK:
        errorMsg = 'A network error caused the video to fail'
        break
      case MediaError.MEDIA_ERR_DECODE:
        errorMsg = 'The video was corrupted or format not supported'
        break
      case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
        errorMsg = 'Video format not supported by browser'
        break
    }
  }

  videoError.value = errorMsg
  isLoading.value = false
  emit('error', new Error(errorMsg))
}

watch(() => props.src, (newSrc) => {
  if (newSrc) {
    loadVideo(newSrc)
  }
})

watch(() => props.autoplay, (autoplay) => {
  if (videoRef.value) {
    if (autoplay) {
      play()
    } else {
      pause()
    }
  }
})

onMounted(() => {
  if (props.src) {
    loadVideo(props.src)
  }

  document.addEventListener('fullscreenchange', handleFullscreenChange)
})

onUnmounted(() => {
  document.removeEventListener('fullscreenchange', handleFullscreenChange)
})

const handleFullscreenChange = () => {
  isFullscreen.value = !!document.fullscreenElement
}

defineExpose({
  play,
  pause,
  togglePlay,
  seek,
  skip,
  loadVideo,
  videoRef,
  currentTime,
  duration,
  isPlaying
})
</script>

<template>
  <div ref="containerRef" class="video-player" :class="{ fullscreen: isFullscreen }">
    <div class="video-wrapper">
      <video
        ref="videoRef"
        class="video-element"
        :muted="isMuted"
        @play="handlePlay"
        @pause="handlePause"
        @ended="handleEnded"
        @timeupdate="handleTimeUpdate"
        @loadedmetadata="handleLoadedMetadata"
        @error="handleError"
        @waiting="isLoading = true"
        @canplay="isLoading = false"
      />

      <div v-if="isLoading" class="video-loading">
        <div class="loading-spinner"></div>
        <span>Loading video...</span>
      </div>

      <div v-if="videoError" class="video-error">
        <span class="error-icon">??</span>
        <span>{{ videoError }}</span>
      </div>

      <div v-if="!props.src && !videoError" class="video-placeholder">
        <span class="placeholder-icon">?</span>
        <span>No video loaded</span>
      </div>

      <canvas ref="canvasRef" class="overlay-canvas" />
    </div>

    <div class="video-controls">
      <div class="progress-container" @click="seek">
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: progress + '%' }"></div>
        </div>
        <div class="progress-thumb" :style="{ left: progress + '%' }"></div>
      </div>

      <div class="controls-row">
        <div class="controls-left">
          <button class="control-btn" @click="togglePlay">
            {{ isPlaying ? '?' : '??' }}
          </button>

          <button class="control-btn" @click="skip(-10)">?</button>
          <button class="control-btn" @click="skip(10)">?</button>

          <div class="volume-control">
            <button class="control-btn" @click="toggleMute">
              {{ isMuted || volume === 0 ? '?' : volume < 0.5 ? '?' : '?' }}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              :value="volume"
              class="volume-slider"
              @input="changeVolume"
            />
          </div>

          <span class="time-display">
            {{ formattedCurrentTime }} / {{ formattedDuration }}
          </span>
        </div>

        <div class="controls-right">
          <select class="speed-select" :value="playbackRate" @change="(e) => setPlaybackRate(parseFloat((e.target as HTMLSelectElement).value))">
            <option value="0.5">0.5x</option>
            <option value="1">1x</option>
            <option value="1.5">1.5x</option>
            <option value="2">2x</option>
          </select>

          <button class="control-btn" @click="toggleFullscreen">
            {{ isFullscreen ? '?' : '?' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.video-player {
  width: 100%;
  display: flex;
  flex-direction: column;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
}

.video-player.fullscreen {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: 9999;
  border-radius: 0;
}

.video-wrapper {
  position: relative;
  flex: 1;
  min-height: 300px;
  background: #000;
}

.video-element {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.overlay-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.video-loading,
.video-error,
.video-placeholder {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  color: rgba(255, 255, 255, 0.7);
}

.video-error {
  color: #ff4444;
}

.placeholder-icon,
.error-icon {
  font-size: 48px;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-top-color: #00d9ff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.video-controls {
  padding: 12px;
  background: rgba(0, 0, 0, 0.8);
}

.progress-container {
  position: relative;
  height: 20px;
  cursor: pointer;
  display: flex;
  align-items: center;
}

.progress-bar {
  width: 100%;
  height: 4px;
  background: rgba(255, 255, 255, 0.3);
  border-radius: 2px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #00d9ff;
  transition: width 0.1s linear;
}

.progress-thumb {
  position: absolute;
  width: 12px;
  height: 12px;
  background: #00d9ff;
  border-radius: 50%;
  transform: translateX(-50%);
  transition: left 0.1s linear;
}

.progress-container:hover .progress-bar {
  height: 6px;
}

.progress-container:hover .progress-thumb {
  width: 16px;
  height: 16px;
}

.controls-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 12px;
}

.controls-left,
.controls-right {
  display: flex;
  align-items: center;
  gap: 8px;
}

.control-btn {
  background: none;
  border: none;
  color: #fff;
  font-size: 18px;
  cursor: pointer;
  padding: 4px 8px;
  border-radius: 4px;
  transition: background 0.2s;
}

.control-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.volume-control {
  display: flex;
  align-items: center;
  gap: 4px;
}

.volume-slider {
  width: 60px;
  height: 4px;
  cursor: pointer;
}

.time-display {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.8);
  font-family: monospace;
  margin-left: 8px;
}

.speed-select {
  background: rgba(255, 255, 255, 0.2);
  border: none;
  color: #fff;
  padding: 4px 8px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.speed-select option {
  background: #1a1a2e;
}
</style>
