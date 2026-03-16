<script setup lang="ts">
import { ref, computed } from 'vue'

interface VideoFile {
  name: string
  file: File
  size: number
  path: string
}

const props = defineProps<{
  accept?: string
}>()

const emit = defineEmits<{
  (e: 'select', files: VideoFile[]): void
}>()

const fileInputRef = ref<HTMLInputElement | null>(null)
const selectedFiles = ref<VideoFile[]>([])
const isLoading = ref(false)
const error = ref<string | null>(null)

const formatSize = (bytes: number): string => {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB'
}

const triggerFileSelect = () => {
  fileInputRef.value?.click()
}

const handleFileSelect = async (event: Event) => {
  const input = event.target as HTMLInputElement
  error.value = null
  isLoading.value = true

  try {
    const files = input.files
    if (!files || files.length === 0) {
      isLoading.value = false
      return
    }

    const aviFiles: VideoFile[] = []

    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      const ext = file.name.toLowerCase().split('.').pop()

      if (props.accept?.includes('.avi') && ext !== 'avi') {
        continue
      }

      aviFiles.push({
        name: file.name,
        file: file,
        size: file.size,
        path: (file as any).webkitRelativePath || file.name
      })
    }

    if (aviFiles.length === 0) {
      error.value = 'No .avi files found in the selected folder'
      isLoading.value = false
      return
    }

    selectedFiles.value = aviFiles
    emit('select', aviFiles)
  } catch (err) {
    error.value = 'Failed to read files: ' + (err as Error).message
  } finally {
    isLoading.value = false
  }
}

const clearSelection = () => {
  selectedFiles.value = []
  if (fileInputRef.value) {
    fileInputRef.value.value = ''
  }
}

const selectedCount = computed(() => selectedFiles.value.length)
</script>

<template>
  <div class="file-selector">
    <input
      ref="fileInputRef"
      type="file"
      :accept="accept || '.avi'"
      webkitdirectory
      multiple
      class="hidden-input"
      @change="handleFileSelect"
    />

    <div class="selector-content" @click="triggerFileSelect">
      <div v-if="isLoading" class="loading-state">
        <span class="loading-spinner"></span>
        <span>Reading files...</span>
      </div>

      <div v-else-if="selectedCount > 0" class="selected-state">
        <div class="file-icon">?</div>
        <div class="file-info">
          <span class="file-count">{{ selectedCount }} .avi file(s)</span>
          <span class="file-hint">Click to select different folder</span>
        </div>
        <button class="clear-btn" @click.stop="clearSelection">?</button>
      </div>

      <div v-else class="empty-state">
        <div class="folder-icon">?</div>
        <span class="instruction">Select Folder with .avi Files</span>
        <span class="hint">Click to browse folder</span>
      </div>
    </div>

    <div v-if="error" class="error-message">
      {{ error }}
    </div>

    <div v-if="selectedCount > 0" class="file-list">
      <div class="file-list-header">
        <span>Selected Files ({{ selectedCount }})</span>
      </div>
      <div class="file-list-content">
        <div
          v-for="(videoFile, index) in selectedFiles.slice(0, 5)"
          :key="index"
          class="file-item"
        >
          <span class="video-icon">?</span>
          <span class="file-name">{{ videoFile.name }}</span>
          <span class="file-size">{{ formatSize(videoFile.size) }}</span>
        </div>
        <div v-if="selectedCount > 5" class="more-files">
          + {{ selectedCount - 5 }} more files
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.file-selector {
  width: 100%;
}

.hidden-input {
  display: none;
}

.selector-content {
  border: 2px dashed rgba(255, 255, 255, 0.3);
  border-radius: 12px;
  padding: 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: rgba(255, 255, 255, 0.05);
}

.selector-content:hover {
  border-color: #00d9ff;
  background: rgba(0, 217, 255, 0.1);
}

.empty-state,
.selected-state,
.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.folder-icon,
.file-icon {
  font-size: 48px;
}

.instruction {
  font-size: 16px;
  font-weight: 500;
  color: #fff;
}

.hint {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.5);
}

.file-count {
  font-size: 16px;
  font-weight: 600;
  color: #00ff88;
}

.file-hint {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.5);
}

.clear-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 24px;
  height: 24px;
  border: none;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.2);
  color: #fff;
  cursor: pointer;
  font-size: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.clear-btn:hover {
  background: rgba(255, 68, 68, 0.8);
}

.loading-spinner {
  width: 24px;
  height: 24px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: #00d9ff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.error-message {
  margin-top: 12px;
  padding: 12px;
  background: rgba(255, 68, 68, 0.2);
  border: 1px solid #ff4444;
  border-radius: 8px;
  color: #ff4444;
  font-size: 14px;
}

.file-list {
  margin-top: 16px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  overflow: hidden;
}

.file-list-header {
  padding: 12px 16px;
  background: rgba(255, 255, 255, 0.1);
  font-size: 14px;
  font-weight: 600;
}

.file-list-content {
  max-height: 200px;
  overflow-y: auto;
  padding: 8px;
}

.file-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px;
  border-radius: 4px;
}

.file-item:hover {
  background: rgba(255, 255, 255, 0.05);
}

.video-icon {
  font-size: 16px;
}

.file-name {
  flex: 1;
  font-size: 13px;
  color: #fff;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-size {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.5);
}

.more-files {
  padding: 8px;
  text-align: center;
  font-size: 12px;
  color: rgba(255, 255, 255, 0.5);
}
</style>
