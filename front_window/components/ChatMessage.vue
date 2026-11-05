<!-- ChatMessage.vue (消息组件) -->
<template>
  <div class="chat-message-list">
    <div v-for="(msg, index) in messages" :key="index" class="message-item">
      <!-- 如果是 AI 回答，进行正则解析渲染 -->
      <div v-if="msg.sender === 'ai'" class="ai-content">
        <!-- 渲染替换后的 HTML -->
        <div v-html="formatAnchorText(msg.text)"></div>
      </div>
      <div v-else class="user-content">{{ msg.text }}</div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted } from 'vue';

const props = defineProps(['messages']);
const emit = defineEmits(['onAnchorClick']);

// 1. 【核心解析函数】：加在这里！
function formatAnchorText(rawText) {
  if (!rawText) return '';
  
  // 使用正则提取 Page 和 BBox 数组，替换为带有 onclick 的 HTML 按钮
  return rawText.replace(
    /\[Anchor:\s*Page\s*(\d+),\s*BBox\(([^)]+)\)\]/g,
    (match, pageNum, bboxStr) => {
      // 构造类似 handleJump(5, [110,400,480,430]) 的内联函数调用
      return `<button 
        class="inline-flex items-center gap-1 px-2 py-0.5 mx-1 my-0.5 text-xs font-medium text-blue-700 bg-blue-100 rounded border border-blue-300 hover:bg-blue-200 cursor-pointer"
        onclick="window.handleGlobalJump(${pageNum}, [${bboxStr}])"
      >
        📍 第 ${pageNum} 页
      </button>`;
    }
  );
}

// 2. 挂载全局跳转方法供 HTML 内联 onclick 调用
onMounted(() => {
  window.handleGlobalJump = (pageNum, bbox) => {
    // 触发父组件/PDF组件的跳转方法
    emit('onAnchorClick', { pageNum, bbox });
  };
});

// 卸载时清理 window 方法，防止内存泄漏
onUnmounted(() => {
  delete window.handleGlobalJump;
});
</script>