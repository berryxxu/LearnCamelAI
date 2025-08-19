from colorama import Fore
import time
import os
from dotenv import load_dotenv

from camel.societies import RolePlaying
from camel.utils import print_text_animated
from camel.models import ModelFactory
from camel.types import ModelPlatformType

# 加载环境变量
load_dotenv(dotenv_path='.env')

def create_model_with_retry(max_retries=3):
    """创建模型实例，带重试机制"""
    api_key = os.getenv('MODELSCOPE_SDK_TOKEN')
    
    if not api_key:
        print(Fore.RED + "错误：未找到 MODELSCOPE_SDK_TOKEN 环境变量")
        return None
    
    for attempt in range(max_retries):
        try:
            print(Fore.YELLOW + f"尝试创建模型实例 (第 {attempt + 1} 次)...")
            
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                model_type="Qwen/Qwen2.5-72B-Instruct",
                url='https://api-inference.modelscope.cn/v1/',
                api_key=api_key,
                # 增加超时设置
                timeout=120,  # 120秒超时
                max_retries=2  # 最大重试次数
            )
            
            print(Fore.GREEN + "模型实例创建成功！")
            return model
            
        except Exception as e:
            print(Fore.RED + f"创建模型实例失败 (第 {attempt + 1} 次): {str(e)}")
            if attempt < max_retries - 1:
                print(Fore.YELLOW + f"等待 5 秒后重试...")
                time.sleep(5)
            else:
                print(Fore.RED + "所有重试都失败了，请检查网络连接和API配置")
                return None

def main(chat_turn_limit=10) -> None:  # 减少对话轮数用于测试
    # 创建模型实例
    model = create_model_with_retry()
    if model is None:
        print(Fore.RED + "无法创建模型实例，程序退出")
        return
    
    task_prompt = "为股票市场开发一个交易机器人"
    
    try:
        print(Fore.CYAN + "正在创建角色扮演会话...")
        role_play_session = RolePlaying(
            assistant_role_name="Python 程序员",
            assistant_agent_kwargs=dict(model=model),
            user_role_name="股票交易员",
            user_agent_kwargs=dict(model=model),
            task_prompt=task_prompt,
            with_task_specify=True,
            task_specify_agent_kwargs=dict(model=model),
            output_language='中文'
        )
        
        print(Fore.GREEN + "角色扮演会话创建成功！")
        
        # 显示系统消息
        print(Fore.GREEN + f"AI 助手系统消息:\n{role_play_session.assistant_sys_msg}\n")
        print(Fore.BLUE + f"AI 用户系统消息:\n{role_play_session.user_sys_msg}\n")
        print(Fore.YELLOW + f"原始任务提示:\n{task_prompt}\n")
        print(Fore.CYAN + f"指定的任务提示:\n{role_play_session.specified_task_prompt}\n")
        print(Fore.RED + f"最终任务提示:\n{role_play_session.task_prompt}\n")
        
        # 开始对话循环
        n = 0
        input_msg = role_play_session.init_chat()
        
        while n < chat_turn_limit:
            n += 1
            print(Fore.MAGENTA + f"\n=== 第 {n} 轮对话 ===")
            
            try:
                assistant_response, user_response = role_play_session.step(input_msg)
                
                # 检查终止条件
                if assistant_response.terminated:
                    print(Fore.GREEN + f"AI 助手已终止。原因: {assistant_response.info['termination_reasons']}.")
                    break
                if user_response.terminated:
                    print(Fore.GREEN + f"AI 用户已终止。原因: {user_response.info['termination_reasons']}.")
                    break
                
                # 显示对话内容
                print_text_animated(Fore.BLUE + f"AI 用户:\n\n{user_response.msg.content}\n")
                print_text_animated(Fore.GREEN + f"AI 助手:\n\n{assistant_response.msg.content}\n")
                
                # 检查是否完成任务
                if "CAMEL_TASK_DONE" in user_response.msg.content:
                    print(Fore.GREEN + "任务完成！")
                    break
                
                input_msg = assistant_response.msg
                
            except Exception as e:
                print(Fore.RED + f"对话过程中发生错误: {str(e)}")
                print(Fore.YELLOW + "尝试继续下一轮对话...")
                time.sleep(2)
                continue
                
    except Exception as e:
        print(Fore.RED + f"创建角色扮演会话时发生错误: {str(e)}")
        return

if __name__ == "__main__":
    print(Fore.CYAN + "=== CAMEL AI 角色扮演系统 ===")
    print(Fore.CYAN + "正在启动...")
    main() 