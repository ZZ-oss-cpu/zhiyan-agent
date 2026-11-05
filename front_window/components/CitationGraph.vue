<!-- CitationGraph.vue -->
<template>
  <div class="w-full h-[500px]" ref="chartRef"></div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';
import * as echarts from 'echarts';

const props = defineProps({
  graphData: {
    type: Object,
    default: () => ({ nodes: [], edges: [] })
  }
});

const emit = defineEmits(['nodeClick']);
const chartRef = ref(null);
let myChart = null;

onMounted(() => {
  myChart = echarts.init(chartRef.value);
  
  // 监听节点点击事件 (点击节点下钻或查看文献详情)
  myChart.on('click', (params) => {
    if (params.dataType === 'node') {
      emit('nodeClick', params.data);
    }
  });

  renderGraph();
});

watch(() => props.graphData, () => renderGraph(), { deep: true });

function renderGraph() {
  if (!myChart || !props.graphData.nodes.length) return;

  const option = {
    title: { text: '局域论文引用图谱 (Cornerstone Identification)', left: 'center' },
    tooltip: {
      formatter: (params) => {
        if (params.dataType === 'node') {
          return `<b>${params.data.id}</b><br/>PageRank 得分: ${params.data.val}`;
        }
        return `${params.data.source} → ${params.data.target}`;
      }
    },
    series: [
      {
        type: 'graph',
        layout: 'force',
        data: props.graphData.nodes.map(n => ({
          id: n.id,
          name: n.label || n.id,
          symbolSize: Math.max(n.val * 20, 15), // 根据 PageRank 权重动态调整大小
          val: n.val,
          itemStyle: { color: n.isCornerstone ? '#f59e0b' : '#3b82f6' } // 基石论文显示为橙色
        })),
        links: props.graphData.edges,
        roam: true, // 支持平移与缩放
        label: { show: true, position: 'right' },
        force: {
          repulsion: 150,
          edgeLength: 80
        },
        lineStyle: {
          color: '#9ca3af',
          curveness: 0.1,
          width: 1.5
        },
        edgeSymbol: ['none', 'arrow'], // 带有箭头的边
        edgeSymbolSize: 8
      }
    ]
  };

  myChart.setOption(option);
}
</script>