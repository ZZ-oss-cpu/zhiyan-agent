// ChatMessage.jsx (React 消息组件)
import React, { useEffect } from 'react';

export function ChatMessage({ messages, onAnchorClick }) {
  useEffect(() => {
    // 绑定 window 级别的跳转回调
    window.handleGlobalJump = (pageNum, bbox) => {
      if (onAnchorClick) {
        onAnchorClick({ pageNum, bbox });
      }
    };
    return () => {
      delete window.handleGlobalJump;
    };
  }, [onAnchorClick]);

  // 【核心解析函数】：加在这里！
  const formatAnchorText = (rawText) => {
    if (!rawText) return '';
    return rawText.replace(
      /\[Anchor:\s*Page\s*(\d+),\s*BBox\(([^)]+)\)\]/g,
      (match, pageNum, bboxStr) => {
        return `<button 
          class="inline-flex items-center px-2 py-0.5 text-xs text-blue-600 bg-blue-50 hover:bg-blue-100 rounded border border-blue-200"
          onclick="window.handleGlobalJump(${pageNum}, [${bboxStr}])"
        >
          📍 第 ${pageNum} 页
        </button>`;
      }
    );
  };

  return (
    <div className="chat-container">
      {messages.map((msg, i) => (
        <div key={i} className="message-box">
          {msg.sender === 'ai' ? (
            <div 
              dangerouslySetInnerHTML={{ __html: formatAnchorText(msg.text) }} 
            />
          ) : (
            <div>{msg.text}</div>
          )}
        </div>
      ))}
    </div>
  );
}