"""
Client script for verifying answers using the vLLM endpoint
"""
import re
import string
import random
import json
import re
import time
import socket
import subprocess
import requests
import os
import threading
from typing import Dict, List, Optional, Union
from openai import OpenAI
import logging
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_thread_local = threading.local()

class GeneralVerifier:
    """自动管理vLLM服务的验证器客户端"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:7832/v1",
                 model_path: str = "xx",
                 model_name: str = "general-verifier",
                 auto_start: bool = True,
                 start_timeout: int = 120,
                 excute_check: bool = False):
        """
        初始化验证器，自动检测并启动vLLM服务
        
        Args:
            base_url: vLLM服务地址
            model_name: 模型名称
            auto_start: 是否自动启动服务
            start_timeout: 启动超时时间（秒）
        """
        self.base_url = base_url
        self.model_name = model_name
        self.port = int(base_url.split(':')[-1].split('/')[0])
        self.vllm_process = None
        self.vllm_log_file = None
        self.model_path = model_path

            
        parsed = urlparse(base_url)
        self.host = parsed.hostname
        self.port = parsed.port
        
        is_local_host = self.host in ("localhost", "127.0.0.1")
        if auto_start:
            if is_local_host:
                if not self._is_service_running():
                    logger.info(f"端口 {self.port} 没有服务，正在启动vLLM...")
                    self._start_vllm_service(timeout=start_timeout)
            else:
                if excute_check and not self._is_service_running():
                    raise RuntimeError(
                        f"远程验证服务未在 {self.base_url} 运行，请先启动该服务或将 base_url 指向本地以允许自动启动。"
                    )
        else:
            if excute_check and not self._is_service_running():
                logger.warning(f"验证服务未运行: {self.base_url}")
        
        self.client = OpenAI(base_url=base_url, api_key="dummy")
    
    def _is_port_open(self) -> bool:
        """检查端口是否开放"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            result = sock.connect_ex((self.host, self.port))
            return result == 0
        except:
            return False
        finally:
            sock.close()
    
    def _is_service_running(self) -> bool:
        """检查vLLM服务是否正在运行"""
        
        endpoints = ["/models"]
        
        for endpoint in endpoints:
            try:
                base = self.base_url.rstrip("/")
                url = f"{base}{endpoint}"
                print(f"Checking health of {url} with proxy: {os.environ.get('http_proxy')}")
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✓ 服务健康检查成功: {url}")
                    return True
            except Exception as e:
                print(f"健康检查失败 {endpoint}: {e}")
                continue
        
        try:
            base = self.base_url.rstrip("/")
            response = requests.get(f"{base}/", timeout=3)
            if response.status_code in [200, 404, 405]:
                print(f"✓ 服务连接成功，状态码: {response.status_code}")
                return True
        except Exception as e:
            print(f"连接测试失败: {e}")
        
        return False
    
    def _start_vllm_service(self, timeout: int = 120):
        """启动vLLM服务"""
        log_dir = os.path.join('logs')
        os.makedirs(log_dir, exist_ok=True)
        
        vllm_log_file = os.path.join(log_dir, f'verifier_vllm_server_{self.port}.log')

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--served-model-name", self.model_name,
        ]
        
        logger.info(f"执行命令: {' '.join(cmd)}")
        logger.info(f"vLLM服务器日志将保存到: {vllm_log_file}")
        
        with open(vllm_log_file, 'w', encoding='utf-8') as log_file:
            log_file.write(f"=== vLLM Server Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.write(f"Model: {self.model_name}\n")
            log_file.write(f"Port: {self.port}\n")
            log_file.write("=" * 60 + "\n\n")
            log_file.flush()
        
        with open(vllm_log_file, 'a', encoding='utf-8') as log_file:
            self.vllm_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
        
        self.vllm_log_file = vllm_log_file
        
        start_time = time.time()
        check_interval = 3
        last_check_time = 0
        
        while time.time() - start_time < timeout:
            current_time = time.time() - start_time
            
            if current_time - last_check_time >= check_interval:
                logger.info(f"等待vLLM服务启动... ({current_time:.1f}s/{timeout}s)")
                last_check_time = current_time
            
            if self._is_service_running():
                logger.info("✓ vLLM服务启动成功！")
                time.sleep(2)
                return
            
            if self.vllm_process.poll() is not None:
                try:
                    with open(self.vllm_log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        error_info = ''.join(lines[-20:])
                except:
                    error_info = "无法读取日志文件"
                raise RuntimeError(f"vLLM启动失败，请查看日志文件: {self.vllm_log_file}\n最后输出:\n{error_info}")
            
            time.sleep(2)
        
        raise TimeoutError(f"vLLM服务在 {timeout} 秒内未能启动，请查看日志文件: {self.vllm_log_file}")
    
    def verify_answer(self, 
                     question: str, 
                     ground_truth: str, 
                     student_answer: str,
                     temperature: float = 0.0) -> Dict:
        """
        验证学生答案是否正确
        
        Args:
            question: 问题
            ground_truth: 标准答案
            student_answer: 学生答案
            temperature: 生成温度
            
        Returns:
            包含验证结果的字典
        """
        prompt = (
            f"User: ### Question: {question}\n\n"
            f"### Ground Truth Answer: {ground_truth}\n\n"
            f"### Student Answer: {student_answer}\n\n"
            "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
            "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
            "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
        )
        
        try:
            completion = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=1024,
                temperature=temperature
            )
            
            response = completion.choices[0].text
            
        except Exception as e:
            error_str = str(e)
            if "maximum context length" in error_str or "requested" in error_str and "tokens" in error_str:
                logger.warning(f"Context length exceeded, truncating input: {error_str}\n\nError Prompt:\n{prompt}")
                
                max_question_length = min(len(question), 1000)
                truncated_question = question[:max_question_length]
                if len(question) > max_question_length:
                    truncated_question += "..."
                
                max_student_length = min(len(student_answer), 1000)
                truncated_student = student_answer[:max_student_length]
                if len(student_answer) > max_student_length:
                    truncated_student += "..."
                
                truncated_prompt = (
                    f"User: ### Question: {truncated_question}\n\n"
                    f"### Ground Truth Answer: {ground_truth}\n\n"
                    f"### Student Answer: {truncated_student}\n\n"
                    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
                    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
                    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
                )
                
                try:
                    completion = self.client.completions.create(
                        model=self.model_name,
                        prompt=truncated_prompt,
                        max_tokens=1024,
                        temperature=temperature
                    )
                    response = completion.choices[0].text
                except Exception as retry_e:
                    logger.error(f"重试后仍然验证失败: {retry_e}")
                    return {
                        "is_correct": None,
                        "error": str(retry_e)
                    }
            else:
                logger.error(f"验证失败: {e}")
                return {
                    "is_correct": None,
                    "error": str(e)
                }
        
        is_correct = None
        if "Final Decision: Yes" in response:
            is_correct = True
        elif "Final Decision: No" in response:
            is_correct = False
        
        return {
            "is_correct": is_correct,
            "response": response
        }
    
    def __del__(self):
        """清理：终止vLLM进程"""
        if self.vllm_process and self.vllm_process.poll() is None:
            logger.info("正在关闭vLLM服务...")
            self.vllm_process.terminate()
            try:
                self.vllm_process.wait(timeout=5)
            except:
                self.vllm_process.kill()


class ScoringVerifier:
    """Combined scoring and verification system."""
    
    def __init__(self, verifier_base_url: str = "http://localhost:7832/v1", 
                 answer_score_weight: float = 1.0):
        """
        Initialize the scoring verifier.
        
        Args:
            verifier_base_url: Base URL for the verifier endpoint
            use_verifier: Whether to use the neural verifier in addition to exact match
            answer_score_weight: Weight for answer correctness
            max_context_length: 最大上下文长度
        """
        self.answer_score_weight = answer_score_weight
        
        
        self.verifier = GeneralVerifier(base_url=verifier_base_url)
    
    def compute_score(self, 
                     solution_str: str, 
                     question: str,
                     ground_truth: Union[str, List[str], Dict],
                     return_details: bool = True,
                     need_extraction: bool = True,
                     verbose: bool = False) -> Dict:
        if isinstance(ground_truth, dict):
            targets = ground_truth.get('target', ground_truth)
        else:
            targets = ground_truth
        
        if isinstance(targets, str):
            targets = [targets]
        
        if solution_str is None:
            if return_details:
                return {
                    'score': 0.,
                    'answer_score': 0.,
                    "format_score": 1.,
                }
            else:
                return 0.
        
        answer_score = 0.
        
        for target in targets:
            result = self.verifier.verify_answer(question, target, solution_str)
            if result.get('is_correct'):
                answer_score = self.answer_score_weight
                break
        
        total_score = answer_score
        
        if verbose:
            print(f"\n[Scoring]")
            print(f"  Question: {question}")
            print(f"  Solution string: {solution_str}")
            print(f"  Ground truth: {targets}")
            print(f"  Answer score: {answer_score}")
            print(f"  Total score: {total_score}")
        
        if return_details:
            return {
                'score': total_score,
                'answer_score': answer_score,
                "format_score": 1.,
            }
        else:
            return total_score




def extract_answer(solution_str: str) -> Optional[str]:
    """
    Extract the answer from the solution string.
    Extract content between <my_answer></my_answer> tags.
    If no match found, return None.
    If multiple matches found, return the last one.
    """
    if solution_str is None:
        return None
    
    pattern = r'<my_answer>(.*?)</my_answer>'
    matches = re.findall(pattern, solution_str, re.DOTALL)
    
    if not matches:
        return None
    
    answer = matches[-1].strip()
    return answer if answer else None


def extract_boxed_content(s: str) -> str | None:
    """
    提取字符串中\\boxed{...}内的内容，多个则返回最后一个，无则返回None
    
    Args:
        s: 待提取的字符串
        
    Returns:
        str | None: 提取到的内容（最后一个）或None
    """
    pattern = r'\\boxed\{([\s\S]*?)\}'
    matches = re.findall(pattern, s)
    
    return matches[-1] if matches else None


def compute_score(solution_str: str, 
                 question: str,
                 ground_truth: Union[str, List[str], Dict],
                 verifier_base_url: str = "http://localhost:7832/v1",
                 answer_score_weight: float = 1.0,
                 return_details: bool = True,
                 need_extraction: bool = False) -> Dict:
    """
    Compute score for a solution string.
    Uses thread-local storage to ensure thread safety in multi-threaded environments.
    """
    if not hasattr(_thread_local, 'verifier') or _thread_local.verifier is None:
        _thread_local.verifier = ScoringVerifier(
            verifier_base_url=verifier_base_url,
            answer_score_weight=answer_score_weight
        )
    scorer = _thread_local.verifier
    
    answer_str = extract_boxed_content(solution_str)
    if not answer_str:
        if return_details:
            return {
                'score': 0.,
                'answer_score': 0.,
                "format_score": 0.,
            }
        else:
            return 0.

    return scorer.compute_score(
        solution_str=answer_str,
        question=question,
        ground_truth=ground_truth,
        return_details=return_details
    )