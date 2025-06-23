#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI聊天模块 - 负责DeepSeek API集成和智能数据分析对话
"""

import streamlit as st
import pandas as pd
import requests
import json
from typing import Dict, List, Any
from config import DEEPSEEK_CONFIG, SYSTEM_TEMPLATE
from data_analysis_tools import DataAnalysisTools, analyze_data_with_tools


class DeepSeekChatModel:
    """
    DeepSeek聊天模型类，封装API调用逻辑
    """
    
    def __init__(self, api_key: str, model_name: str = None):
        """
        初始化DeepSeek聊天模型
        
        Args:
            api_key: DeepSeek API密钥
            model_name: 模型名称，默认使用配置中的模型
        """
        self.api_key = api_key
        self.model_name = model_name or DEEPSEEK_CONFIG['default_model']
        self.temperature = DEEPSEEK_CONFIG['temperature']
        self.api_url = DEEPSEEK_CONFIG['api_url']
    
    def call_api(self, messages: List[Dict[str, str]]) -> str:
        """
        调用DeepSeek API
        
        Args:
            messages: 消息列表
            
        Returns:
            str: API响应内容
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature
            }
            
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            return f"API调用失败：{str(e)}"
        except KeyError as e:
            return f"响应格式错误：{str(e)}"
        except Exception as e:
            return f"处理出错：{str(e)}"
    
    def test_api_connection(self) -> Dict[str, Any]:
        """
        测试API连接是否正常
        
        Returns:
            Dict[str, Any]: 包含连接状态和消息的字典
                - success: bool, 连接是否成功
                - message: str, 状态消息
                - response_time: float, 响应时间（秒）
        """
        import time
        
        try:
            start_time = time.time()
            
            # 发送一个简单的测试消息
            test_messages = [
                {"role": "user", "content": "你好，这是一个连接测试。请简单回复'连接成功'。"}
            ]
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": test_messages,
                "temperature": 0.1,
                "max_tokens": 50
            }
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    return {
                        "success": True,
                        "message": "✅ API连接成功！响应正常。",
                        "response_time": round(response_time, 2)
                    }
                else:
                    return {
                        "success": False,
                        "message": "❌ API响应格式异常，请检查API配置。",
                        "response_time": round(response_time, 2)
                    }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "message": "❌ API密钥无效，请检查您的API Key。",
                    "response_time": round(response_time, 2)
                }
            elif response.status_code == 429:
                return {
                    "success": False,
                    "message": "❌ API调用频率超限，请稍后再试。",
                    "response_time": round(response_time, 2)
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ API调用失败，状态码: {response.status_code}",
                    "response_time": round(response_time, 2)
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "message": "❌ 连接超时，请检查网络连接。",
                "response_time": 30.0
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "message": "❌ 网络连接错误，请检查网络设置。",
                "response_time": 0.0
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ 连接测试失败: {str(e)}",
                "response_time": 0.0
            }
    
    def analyze_data(self, user_input: str, data: pd.DataFrame, chat_history: List[Dict[str, str]] = None) -> str:
        """
        分析数据并生成响应（支持多轮对话）
        
        Args:
            user_input: 用户输入
            data: 数据框
            chat_history: 聊天历史记录
            
        Returns:
            str: 分析结果
        """
        # 智能选择并执行分析
        analysis_results = self._select_and_execute_analysis(user_input, data)
        
        # 构建包含实际分析结果的提示
        columns_info = ', '.join(data.columns.tolist()[:10])
        if len(data.columns) > 10:
            columns_info += f"... (共{len(data.columns)}列)"
        
        data_info = f"""数据概览：
        - 行数：{len(data)}
        - 列数：{len(data.columns)}
        - 主要列名：{columns_info}
        
        实际分析结果：
        {json.dumps(analysis_results, ensure_ascii=False, indent=2)}
        
        用户问题：{user_input}
        
        请基于以上实际分析结果回答用户问题，提供专业的数据洞察和解释。如果用户提到了之前的对话内容，请结合历史对话上下文进行回答。"""
        
        # 构建包含历史对话的消息列表
        messages = self._build_conversation_messages(data_info, chat_history)
        
        return self.call_api(messages)
    
    def _build_conversation_messages(self, data_info: str, chat_history: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """
        构建包含历史对话的消息列表
        
        Args:
            data_info: 数据信息和当前问题
            chat_history: 聊天历史记录
            
        Returns:
            List[Dict[str, str]]: 消息列表
        """
        # 系统提示
        system_message = {
            "role": "system", 
            "content": "你是一个专业的数据分析助手。你已经获得了实际的数据分析结果，请基于这些具体的数值和统计信息来回答用户问题，提供专业、准确的数据洞察。你能够记住之前的对话内容，并在回答时结合历史上下文。请用中文回答。"
        }
        
        messages = [system_message]
        
        # 添加历史对话上下文（限制长度避免超出API限制）
        if chat_history:
            # 只保留最近的对话记录，避免上下文过长
            recent_history = self._trim_chat_history(chat_history)
            
            for message in recent_history:
                if message['role'] in ['user', 'assistant']:
                    messages.append({
                        "role": message['role'],
                        "content": message['content']
                    })
        
        # 添加当前用户问题和数据信息
        messages.append({"role": "user", "content": data_info})
        
        return messages
    
    def _trim_chat_history(self, chat_history: List[Dict[str, str]], max_messages: int = 10) -> List[Dict[str, str]]:
        """
        修剪聊天历史，保留最近的对话记录
        
        Args:
            chat_history: 完整的聊天历史
            max_messages: 最大保留的消息数量
            
        Returns:
            List[Dict[str, str]]: 修剪后的聊天历史
        """
        if not chat_history or len(chat_history) <= max_messages:
            return chat_history or []
        
        # 保留最近的对话，确保用户和助手的对话成对出现
        recent_history = chat_history[-max_messages:]
        
        # 如果第一条消息是助手回复，则去掉它以保持对话的完整性
        if recent_history and recent_history[0].get('role') == 'assistant':
            recent_history = recent_history[1:]
        
        return recent_history
    
    def _select_and_execute_analysis(self, user_input: str, data: pd.DataFrame) -> Dict:
        """
        根据用户输入智能选择并执行相应的数据分析
        
        Args:
            user_input: 用户输入
            data: 数据框
            
        Returns:
            Dict: 分析结果
        """
        user_input_lower = user_input.lower()
        
        # 根据关键词选择分析类型
        if any(keyword in user_input_lower for keyword in ['相关', '关联', 'correlation', '相关性']):
            return analyze_data_with_tools(data, 'correlation')
        elif any(keyword in user_input_lower for keyword in ['异常', '离群', 'outlier', '异常值']):
            return analyze_data_with_tools(data, 'outliers')
        elif any(keyword in user_input_lower for keyword in ['趋势', 'trend', '变化', '发展']):
            return analyze_data_with_tools(data, 'trends')
        elif any(keyword in user_input_lower for keyword in ['分布', 'distribution', '正态', '偏度']):
            return analyze_data_with_tools(data, 'distribution')
        elif any(keyword in user_input_lower for keyword in ['统计', 'statistics', '描述', '概况']):
            return analyze_data_with_tools(data, 'basic')
        else:
            # 默认进行综合分析
            return analyze_data_with_tools(data, 'comprehensive')


class ChatProcessor:
    """
    聊天处理器类，负责处理用户输入和生成响应
    """
    
    def __init__(self):
        """
        初始化聊天处理器
        """
        self.deepseek_model = None
    
    @staticmethod
    def test_api_key(api_key: str, model_name: str = None) -> Dict[str, Any]:
        """
        测试API密钥是否有效
        
        Args:
            api_key: API密钥
            model_name: 模型名称，默认使用配置中的模型
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        if not api_key or api_key.strip() == "":
            return {
                "success": False,
                "message": "❌ 请输入API密钥",
                "response_time": 0.0
            }
        
        try:
            # 创建临时的DeepSeek模型实例进行测试
            temp_model = DeepSeekChatModel(api_key.strip(), model_name)
            return temp_model.test_api_connection()
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ 测试过程中出现错误: {str(e)}",
                "response_time": 0.0
            }
    
    def setup_deepseek_model(self, api_key: str, model_name: str = None) -> bool:
        """
        设置DeepSeek模型
        
        Args:
            api_key: API密钥
            model_name: 模型名称
            
        Returns:
            bool: 设置是否成功
        """
        try:
            self.deepseek_model = DeepSeekChatModel(api_key, model_name)
            return True
        except Exception as e:
            st.error(f"设置DeepSeek模型失败: {str(e)}")
            return False
    
    def process_chat_input(self, user_input: str, data: pd.DataFrame, 
                          deepseek_api_key: str = None, deepseek_model: str = None,
                          chat_history: List[Dict[str, str]] = None) -> Dict[str, str]:
        """
        处理聊天输入
        
        Args:
            user_input: 用户输入
            data: 数据框
            deepseek_api_key: DeepSeek API密钥
            deepseek_model: DeepSeek模型名称
            
        Returns:
            Dict[str, str]: 包含角色和内容的响应字典
        """
        response = {'role': 'assistant', 'content': ''}
        
        try:
            # 检查API密钥
            if not deepseek_api_key:
                response['content'] = "❌ 请在侧边栏输入DeepSeek API密钥后再使用AI分析功能。"
                return response
            
            # 设置并使用DeepSeek模型
            if self.setup_deepseek_model(deepseek_api_key, deepseek_model):
                # 传递聊天历史以支持多轮对话
                api_response = self.deepseek_model.analyze_data(user_input, data, chat_history)
                response['content'] = api_response
            else:
                response['content'] = "DeepSeek模型设置失败，请检查配置。"
                
        except Exception as e:
            response['content'] = f"处理出错：{str(e)}"
        
        return response


class ChatHistoryManager:
    """
    聊天历史管理器
    """
    
    @staticmethod
    def initialize_chat_history():
        """
        初始化聊天历史
        """
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    @staticmethod
    def add_message(role: str, content: str):
        """
        添加消息到聊天历史
        
        Args:
            role: 消息角色（user/assistant）
            content: 消息内容
        """
        st.session_state.chat_history.append({
            'role': role,
            'content': content
        })
    
    @staticmethod
    def clear_history():
        """
        清空聊天历史
        """
        st.session_state.chat_history = []
    
    @staticmethod
    def display_chat_history():
        """
        显示聊天历史
        """
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"**🙋 用户:** {message['content']}")
            else:
                st.markdown(f"**🤖 助手:** {message['content']}")