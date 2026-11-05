<!-- PdfViewer.vue -->
<template>
  <div class="pdf-container relative overflow-auto h-full border-r">
    <div ref="pdfWrapper" class="relative inline-block">
      <!-- PDF 渲染 Canvas -->
      <canvas ref="pdfCanvas"></canvas>
      <!-- 高亮框 叠加层 -->
      <div 
        v-if="highlightStyle.visible"
        class="absolute border-2 border-red-500 bg-yellow-200/40 transition-all duration-300 pointer-events-none"
        :style="highlightStyle.style"
      ></div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, reactive } from 'vue';
import * as pdfjsLib from 'pdfjs-dist';

// 配置 PDF.js Worker
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

const props = defineProps(['pdfUrl']);
const pdfCanvas = ref(null);
const pdfWrapper = ref(null);
let pdfDoc = null;
let currentScale = 1.5;

const highlightStyle = reactive({
  visible: false,
  style: {}
});

// 1. 初始化并加载 PDF
onMounted(async () => {
  if (props.pdfUrl) {
    pdfDoc = await pdfjsLib.getDocument(props.pdfUrl).promise;
    renderPage(1); // 默认渲染第 1 页
  }
});

// 2. 渲染指定页码
async function renderPage(pageNum) {
  const page = await pdfDoc.getPage(pageNum);
  const viewport = page.getViewport({ scale: currentScale });
  const canvas = pdfCanvas.value;
  const context = canvas.getContext('2d');

  canvas.height = viewport.height;
  canvas.width = viewport.width;

  await page.render({ canvasContext: context, viewport }).promise;
  return viewport;
}

// 3. 暴露给外部调用的锚点跳转与高亮函数
async function jumpToAnchor(pageNum, bbox) {
  // 渲染目标页
  const viewport = await renderPage(pageNum);

  // 解析 bbox 坐标 [x1, y1, x2, y2] 并根据缩放比例计算 CSS 像素
  const [x1, y1, x2, y2] = bbox;
  
  // PDF.js 坐标系转换 (PDF 坐标系原点通常在左下角或左上角，视解析器而定)
  const rect = viewport.convertToViewportRectangle([x1, y1, x2, y2]);
  const width = Math.abs(rect[2] - rect[0]);
  const height = Math.abs(rect[3] - rect[1]);
  const left = Math.min(rect[0], rect[2]);
  const top = Math.min(rect[1], rect[3]);

  // 设置高亮框样式
  highlightStyle.style = {
    left: `${left}px`,
    top: `${top}px`,
    width: `${width}px`,
    height: `${height}px`
  };
  highlightStyle.visible = true;

  // 滚动至高亮区域
  pdfWrapper.value?.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

defineExpose({ jumpToAnchor });
</script>