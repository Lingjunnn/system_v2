<template>
  <div class="monitor-view">
    <header class="monitor-header">
      <div class="header-left">
        <h1 class="system-title">Coal Mine Safety Monitoring System</h1>
      </div>
      <div class="header-right">
        <div class="connection-status" :class="{ connected: wsConnected }">
          <span class="status-dot"></span>
          <span>{{ wsConnected ? 'Connected' : 'Disconnected' }}</span>
        </div>
        <div class="current-time">{{ currentTime }}</div>
      </div>
    </header>

    <main class="monitor-main">
      <div class="main-content">
        <div class="video-container" ref="videoContainerRef">
          <video
            ref="videoRef"
            class="video-element"
            playsinline
            muted
          ></video>
          <canvas ref="canvasRef" class="overlay-canvas"></canvas>

          <div v-if="!store.videoSource && !currentVideoFile && !isLoadingVideos" class="video-placeholder">
            <span class="placeholder-icon">?</span>
            <p>Select a folder containing video files to start monitoring</p>
            <p class="hint">Use the folder selector on the right side</p>
          </div>

          <div v-if="isLoadingVideos" class="model-loading">
            <div class="loading-spinner"></div>
            <span>Loading videos...</span>
          </div>

          <div v-if="lastResult" class="detection-result" :class="{ fighting: lastResult.isFight }">
            <div class="result-badge">
              {{ lastResult.isFight ? '? FIGHT DETECTED' : '? NORMAL' }}
            </div>
            <div class="result-confidence">
              Confidence: {{ (lastResult.score * 100).toFixed(1) }}%
            </div>
            <div class="result-inference">
              Inference: {{ lastResult.inferenceTimeMs.toFixed(1) }}ms
            </div>
          </div>

          <div class="buffer-indicator" v-if="wsConnected">
            <span>Buffer: {{ (bufferProgress * 100).toFixed(0) }}%</span>
            <div class="buffer-bar-mini">
              <div class="buffer-fill-mini" :style="{ width: (bufferProgress * 100) + '%' }"></div>
            </div>
          </div>
        </div>

        <div class="playlist-controls" v-if="currentVideoFiles.length > 0">
          <div class="playlist-info">
            <span class="playlist-name">{{ currentVideoFiles[currentVideoIndex]?.name || '' }}</span>
            <span class="playlist-position">{{ currentVideoIndex + 1 }} / {{ currentVideoFiles.length }}</span>
          </div>
          <div class="playlist-bar" @click="handlePlaylistSeek">
            <div class="playlist-progress" :style="{ width: playlistProgress + '%' }"></div>
          </div>
        </div>
      </div>

      <aside class="sidebar">
        <el-card class="status-card">
          <template #header>
            <div class="card-header">
              <span>System Status</span>
              <el-tag :type="statusTagType" size="small">{{ statusText }}</el-tag>
            </div>
          </template>
          <div class="status-content">
            <div class="status-indicator" :class="statusClass">
              <span class="status-dot"></span>
              <span class="status-text">{{ statusText }}</span>
            </div>
            <div class="fps-display">
              <span class="fps-label">Processing FPS:</span>
              <span class="fps-value">{{ processingFps }}</span>
            </div>
            <div class="model-status">
              <span class="model-label">Backend:</span>
              <span :class="['model-state', { loaded: wsConnected }]">
                {{ wsConnected ? '? Connected' : '? Not Connected' }}
              </span>
            </div>
          </div>
        </el-card>

        <el-card class="control-card">
          <template #header>
            <div class="card-header">
              <span>Video Source</span>
            </div>
          </template>

          <div class="folder-selector">
            <input
              ref="folderInputRef"
              type="file"
              accept="video/*"
              webkitdirectory
              multiple
              style="display: none;"
              @change="handleFolderSelect"
            />
            <div class="folder-selector-content" @click="folderInputRef?.click()">
              <div class="folder-icon">?</div>
              <div class="folder-info">
                <span class="folder-label">Select Video Folder</span>
                <span class="folder-hint">Click to browse folder with .mp4 videos</span>
              </div>
            </div>
          </div>

          <div v-if="currentVideoFiles.length > 0" class="video-playlist">
            <div class="playlist-header">
              <span>Playlist ({{ currentVideoIndex + 1 }}/{{ currentVideoFiles.length }})</span>
              <el-button link type="danger" size="small" @click="stopVideo">{{ isPaused ? 'Resume' : 'Stop' }}</el-button>
            </div>
            <div class="playlist-items">
              <div
                v-for="(file, index) in currentVideoFiles"
                :key="index"
                class="playlist-item"
                :class="{ active: currentVideoIndex === index }"
                @click="playVideo(index)"
              >
                <span class="video-icon">?</span>
                <span class="video-name">{{ file.name }}</span>
              </div>
            </div>
          </div>
        </el-card>

        <el-card class="log-card">
          <template #header>
            <div class="card-header">
              <span>Alarm Logs</span>
              <el-button
                v-if="store.alarmLogs.length > 0"
                link
                type="danger"
                size="small"
                @click="store.clearAlarmLogs"
              >
                Clear
              </el-button>
            </div>
          </template>
          <div class="log-list" ref="logListRef">
            <div
              v-for="log in store.alarmLogs"
              :key="log.id"
              class="log-item"
              :class="log.type"
            >
              <div class="log-header">
                <el-tag :type="log.type === 'critical' ? 'danger' : 'warning'" size="small">
                  {{ log.type.toUpperCase() }}
                </el-tag>
                <span class="log-time">{{ log.timestamp }}</span>
              </div>
              <div class="log-message">{{ log.message }}</div>
            </div>
            <el-empty v-if="store.alarmLogs.length === 0" description="No alarms" :image-size="60" />
          </div>
        </el-card>
      </aside>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useMonitorStore } from '@/stores/monitorStore'
import { useWebSocketDetection, WebSocketDetectionResult } from '@/composables/useWebSocketDetection'
import { ElMessage } from 'element-plus'

interface VideoFile {
  name: string
  url: string
}

const store = useMonitorStore()

const videoRef = ref<HTMLVideoElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const videoContainerRef = ref<HTMLDivElement | null>(null)
const folderInputRef = ref<HTMLInputElement | null>(null)

const currentTime = ref('')
const currentVideoFile = ref<string | null>(null)
const currentVideoFiles = ref<VideoFile[]>([])
const currentVideoIndex = ref(0)
const isLoadingVideos = ref(false)
const isPaused = ref(false)
const processingFps = ref(0)
const lastResult = ref<WebSocketDetectionResult | null>(null)
const bufferProgress = ref(0)

let timeTimer: number | null = null
let detectionTimer: number | null = null
let fpsCounter = 0
let fpsTimer: number | null = null

const wsUrl = 'ws://localhost:8000/ws/detect'

const {
  isConnected: wsConnected,
  connect: wsConnect,
  disconnect: wsDisconnect,
  startContinuousDetection: startWsDetection,
  stopContinuousDetection: stopWsDetection,
  reset: wsReset
} = useWebSocketDetection({
  url: wsUrl,
  inferenceFps: 20,
  frameWidth: 640,
  frameHeight: 640,
  frameQuality: 0.85,
  onDetection: (result) => {
    lastResult.value = result
    bufferProgress.value = 1.0

    fpsCounter++
    updateFps()

    if (result.isFight) {
      store.setFighting(true)
      store.addAlarmLog('critical', `Fighting detected! Confidence: ${(result.score * 100).toFixed(1)}%`)
    } else {
      store.setFighting(false)
    }

    drawResult(result)
  },
  onConnect: () => {
    console.log('[UI] WebSocket connected')
  },
  onDisconnect: () => {
    console.log('[UI] WebSocket disconnected')
  }
})

const statusText = computed(() => {
  if (lastResult.value?.isFight) return 'Critical'
  if (currentVideoFile.value) return 'Normal'
  return 'Idle'
})

const statusTagType = computed(() => {
  if (lastResult.value?.isFight) return 'danger'
  if (currentVideoFile.value) return 'success'
  return 'info'
})

const statusClass = computed(() => {
  if (lastResult.value?.isFight) return 'critical'
  if (currentVideoFile.value) return 'normal'
  return 'idle'
})

const playlistProgress = computed(() => {
  if (!videoRef.value || !currentVideoFile.value) return 0
  return (videoRef.value.currentTime / videoRef.value.duration) * 100
})

const updateTime = () => {
  const now = new Date()
  currentTime.value = now.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}

const updateFps = () => {
  if (fpsTimer) {
    clearTimeout(fpsTimer)
  }
  fpsTimer = window.setTimeout(() => {
    processingFps.value = fpsCounter
    fpsCounter = 0
  }, 1000)
}

const handleFolderSelect = async (event: Event) => {
  const input = event.target as HTMLInputElement
  const files = input.files

  if (!files || files.length === 0) return

  isLoadingVideos.value = true
  stopVideo()

  currentVideoFiles.value = []
  const videoExtensions = ['.mp4', '.webm', '.mkv', '.mov', '.avi']

  for (let i = 0; i < files.length; i++) {
    const file = files[i]
    const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'))

    if (videoExtensions.includes(ext)) {
      currentVideoFiles.value.push({
        name: file.name,
        url: URL.createObjectURL(file)
      })
    }
  }

  if (currentVideoFiles.value.length === 0) {
    ElMessage.warning('No video files found in the selected folder')
    isLoadingVideos.value = false
    return
  }

  currentVideoFiles.value.sort((a, b) => a.name.localeCompare(b.name))

  isLoadingVideos.value = false
  ElMessage.success(`Loaded ${currentVideoFiles.value.length} video(s)`)

  await playVideo(0)

  if (folderInputRef.value) {
    const newInput = document.createElement('input')
    newInput.type = 'file'
    newInput.accept = 'video/*'
    newInput.webkitdirectory = true
    newInput.multiple = true
    newInput.style.display = 'none'
    newInput.onchange = handleFolderSelect as any
    input.parentNode?.replaceChild(newInput, input)
    folderInputRef.value = newInput
  }
}

const playVideo = async (index: number) => {
  if (index < 0 || index >= currentVideoFiles.value.length) return

  isPaused.value = false

  if (videoRef.value) {
    videoRef.value.pause()
    if (videoRef.value.src) {
      const oldUrl = videoRef.value.src
      videoRef.value.removeAttribute('src')
      URL.revokeObjectURL(oldUrl)
    }
    videoRef.value.onended = null
    videoRef.value.onerror = null
  }

  currentVideoIndex.value = index
  const videoFile = currentVideoFiles.value[index]

  if (!videoRef.value) return

  videoRef.value.src = videoFile.url
  currentVideoFile.value = videoFile.url
  store.setVideoSource('folder')

  videoRef.value.onloadeddata = async () => {
    if (videoRef.value) {
      try {
        await videoRef.value.play()
        store.setFps(30)
        startVideoDetection()
      } catch (e) {
        console.error('[UI] Play error:', e)
      }
    }
  }

  videoRef.value.onended = () => {
    if (currentVideoIndex.value < currentVideoFiles.value.length - 1) {
      playNextVideo()
    } else {
      ElMessage.info('Playlist finished')
    }
  }

  videoRef.value.onerror = () => {
    ElMessage.error(`Failed to load video: ${videoFile.name}`)
    if (currentVideoIndex.value < currentVideoFiles.value.length - 1) {
      playNextVideo()
    }
  }
}

const playNextVideo = () => {
  if (currentVideoIndex.value < currentVideoFiles.value.length - 1) {
    playVideo(currentVideoIndex.value + 1)
  }
}

const handlePlaylistSeek = (event: MouseEvent) => {
  if (!videoRef.value) return

  const target = event.currentTarget as HTMLElement
  const rect = target.getBoundingClientRect()
  const pos = (event.clientX - rect.left) / rect.width
  videoRef.value.currentTime = pos * videoRef.value.duration
}

const stopVideo = () => {
  if (isPaused.value) {
    resumeVideo()
    return
  }

  isPaused.value = true
  stopVideoDetection()

  if (videoRef.value) {
    videoRef.value.pause()
  }

  store.setFighting(false)
  lastResult.value = null
  bufferProgress.value = 0

  if (canvasRef.value) {
    const ctx = canvasRef.value.getContext('2d')
    if (ctx) {
      ctx.clearRect(0, 0, canvasRef.value.width, canvasRef.value.height)
    }
  }

  wsDisconnect()
  wsReset()
}

const resumeVideo = async () => {
  if (videoRef.value && currentVideoFile.value) {
    if (!wsConnected.value) {
      try {
        await wsConnect()
      } catch (e) {
        ElMessage.error('Cannot connect to backend server')
        return
      }
    }
    videoRef.value.play()
    isPaused.value = false
    startVideoDetection()
  }
}

const startVideoDetection = async () => {
  if (detectionTimer !== null) {
    return
  }

  if (!wsConnected.value) {
    console.log('[UI] Connecting to WebSocket server...')
    try {
      await wsConnect()
    } catch (e) {
      console.error('[UI] Failed to connect to WebSocket:', e)
      ElMessage.error('Cannot connect to backend server. Please ensure the server is running.')
      return
    }
  }

  if (videoRef.value) {
    wsReset()
    const currentVideoName = currentVideoFiles.value[currentVideoIndex.value]?.name || 'unknown.avi'
    startWsDetection(videoRef.value, currentVideoName)
    detectionTimer = 1
  }
}

const stopVideoDetection = () => {
  if (detectionTimer) {
    cancelAnimationFrame(detectionTimer)
    detectionTimer = null
  }
  stopWsDetection()
}

const drawResult = (result: WebSocketDetectionResult) => {
  if (!canvasRef.value) return

  const ctx = canvasRef.value.getContext('2d')
  if (!ctx) return

  ctx.clearRect(0, 0, canvasRef.value.width, canvasRef.value.height)

  const color = result.isFight ? '#ff4444' : '#00ff88'
  const label = result.isFight ? '? FIGHT DETECTED' : '? NORMAL'
  const confidence = (result.score * 100).toFixed(1)

  ctx.strokeStyle = color
  ctx.lineWidth = 4
  ctx.strokeRect(20, 20, canvasRef.value.width - 40, canvasRef.value.height - 40)

  ctx.fillStyle = color
  ctx.font = 'bold 28px Arial'
  ctx.fillText(`${label}`, 30, 60)
  ctx.font = '20px Arial'
  ctx.fillText(`Confidence: ${confidence}%`, 30, 90)

  if (result.isFight) {
    ctx.fillStyle = 'rgba(255, 0, 0, 0.15)'
    ctx.fillRect(0, 0, canvasRef.value.width, canvasRef.value.height)
  }
}

const resizeCanvas = () => {
  if (videoContainerRef.value && canvasRef.value) {
    const container = videoContainerRef.value
    canvasRef.value.width = container.clientWidth
    canvasRef.value.height = container.clientHeight
  }
}

watch(currentVideoFile, (newVal) => {
  if (newVal) {
    setTimeout(resizeCanvas, 100)
  }
})

onMounted(() => {
  updateTime()
  timeTimer = window.setInterval(updateTime, 1000)
  window.addEventListener('resize', resizeCanvas)

  setTimeout(resizeCanvas, 100)
})

onUnmounted(() => {
  if (timeTimer) {
    clearInterval(timeTimer)
  }

  if (fpsTimer) {
    clearTimeout(fpsTimer)
  }

  stopVideoDetection()
  stopVideo()

  currentVideoFiles.value.forEach(f => URL.revokeObjectURL(f.url))

  window.removeEventListener('resize', resizeCanvas)
})
</script>

<style scoped>
.monitor-view {
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #fff;
}

.monitor-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  background: rgba(0, 0, 0, 0.3);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.header-left {
  display: flex;
  align-items: center;
}

.system-title {
  font-size: 24px;
  font-weight: 600;
  background: linear-gradient(90deg, #00d9ff, #00ff88);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 24px;
}

.connection-status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: #ff4444;
}

.connection-status.connected {
  color: #00ff88;
}

.connection-status .status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #ff4444;
}

.connection-status.connected .status-dot {
  background: #00ff88;
}

.current-time {
  font-size: 18px;
  font-family: 'Courier New', monospace;
  color: #00d9ff;
}

.monitor-main {
  flex: 1;
  display: flex;
  padding: 16px;
  gap: 16px;
  overflow: hidden;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.video-container {
  position: relative;
  flex: 1;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
  min-height: 400px;
}

.video-element {
  position: absolute;
  top: 0;
  left: 0;
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

.video-placeholder {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  color: #666;
}

.placeholder-icon {
  font-size: 64px;
  margin-bottom: 16px;
}

.hint {
  font-size: 12px;
  color: #444;
  margin-top: 8px;
}

.model-loading {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  color: #00d9ff;
  z-index: 10;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(0, 217, 255, 0.3);
  border-top-color: #00d9ff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.detection-result {
  position: absolute;
  top: 16px;
  left: 16px;
  padding: 12px 20px;
  background: rgba(0, 255, 136, 0.9);
  border-radius: 8px;
  color: #000;
  font-weight: 600;
  z-index: 5;
}

.detection-result.fighting {
  background: rgba(255, 68, 68, 0.9);
  color: #fff;
  animation: blink 0.5s infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

.result-confidence {
  font-size: 12px;
  margin-top: 4px;
}

.result-inference {
  font-size: 11px;
  margin-top: 2px;
  opacity: 0.8;
}

.buffer-indicator {
  position: absolute;
  bottom: 16px;
  right: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #00d9ff;
  background: rgba(0, 0, 0, 0.6);
  padding: 6px 12px;
  border-radius: 4px;
}

.buffer-bar-mini {
  width: 60px;
  height: 4px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 2px;
  overflow: hidden;
}

.buffer-fill-mini {
  height: 100%;
  background: #00d9ff;
  transition: width 0.3s ease;
}

.playlist-controls {
  padding: 12px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  margin-top: 8px;
}

.playlist-info {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  margin-bottom: 8px;
}

.playlist-name {
  color: #00d9ff;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 70%;
}

.playlist-position {
  color: #888;
}

.playlist-bar {
  height: 4px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 2px;
  cursor: pointer;
  overflow: hidden;
}

.playlist-progress {
  height: 100%;
  background: linear-gradient(90deg, #00d9ff, #00ff88);
  transition: width 0.1s linear;
}

.sidebar {
  width: 360px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.status-card,
.control-card,
.log-card {
  background: rgba(255, 255, 255, 0.05) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  color: #fff;
}

.status-card :deep(.el-card__header),
.control-card :deep(.el-card__header),
.log-card :deep(.el-card__header) {
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  background: transparent;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
}

.status-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  border-radius: 8px;
  background: rgba(0, 0, 0, 0.3);
}

.status-indicator.normal {
  border-left: 4px solid #00ff88;
}

.status-indicator.critical {
  border-left: 4px solid #ff4444;
}

.status-indicator.idle {
  border-left: 4px solid #888;
}

.status-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #00ff88;
}

.status-indicator.critical .status-dot {
  background: #ff4444;
  animation: pulse 1s infinite;
}

.status-indicator.idle .status-dot {
  background: #888;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.status-text {
  font-size: 16px;
  font-weight: 600;
}

.fps-display {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
}

.fps-label {
  color: #888;
}

.fps-value {
  font-size: 20px;
  font-weight: 700;
  color: #00d9ff;
}

.model-status {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  font-size: 12px;
}

.model-label {
  color: #888;
}

.model-state {
  color: #ff4444;
}

.model-state.loaded {
  color: #00ff88;
}

.folder-selector {
  margin-bottom: 16px;
}

.folder-selector-content {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  border: 2px dashed rgba(255, 255, 255, 0.3);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  background: rgba(255, 255, 255, 0.05);
}

.folder-selector-content:hover {
  border-color: #00d9ff;
  background: rgba(0, 217, 255, 0.1);
}

.folder-icon {
  font-size: 32px;
}

.folder-label {
  font-size: 14px;
  font-weight: 500;
}

.folder-hint {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.5);
}

.video-playlist {
  max-height: 250px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.playlist-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  color: #888;
  padding: 8px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.playlist-items {
  flex: 1;
  overflow-y: auto;
  margin-top: 8px;
}

.playlist-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px;
  cursor: pointer;
  border-radius: 4px;
  font-size: 12px;
}

.playlist-item:hover {
  background: rgba(255, 255, 255, 0.1);
}

.playlist-item.active {
  background: rgba(0, 217, 255, 0.2);
  color: #00d9ff;
}

.video-icon {
  font-size: 14px;
}

.video-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.log-card {
  flex: 1;
  min-height: 200px;
}

.log-card :deep(.el-card__body) {
  max-height: 300px;
  overflow: hidden;
}

.log-list {
  max-height: 250px;
  overflow-y: auto;
}

.log-list::-webkit-scrollbar {
  width: 6px;
}

.log-list::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.3);
  border-radius: 3px;
}

.log-list::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.2);
  border-radius: 3px;
}

.log-item {
  padding: 12px;
  margin-bottom: 8px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  border-left: 3px solid transparent;
}

.log-item.warning {
  border-left-color: #ffaa00;
}

.log-item.critical {
  border-left-color: #ff4444;
}

.log-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.log-time {
  font-size: 12px;
  color: #888;
}

.log-message {
  font-size: 14px;
}

.log-card :deep(.el-empty) {
  padding: 20px 0;
}
</style>
