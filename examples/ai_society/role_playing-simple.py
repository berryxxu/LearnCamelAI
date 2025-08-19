from colorama import Fore
import os
from dotenv import load_dotenv

from camel.societies import RolePlaying
from camel.utils import print_text_animated
from camel.models import ModelFactory
from camel.types import ModelPlatformType

# 加载环境变量
load_dotenv(dotenv_path='.env')

def create_simple_model():
    """创建一个简单的模型实例，使用更小的模型"""
    api_key = os.getenv('MODELSCOPE_SDK_TOKEN')
    
    if not api_key:
        print(Fore.RED + "错误：未找到 MODELSCOPE_SDK_TOKEN 环境变量")
        return None
    
    try:
        print(Fore.YELLOW + "正在创建模型实例...")
        
        # 尝试使用更小的模型
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="Qwen/Qwen2.5-7B-Instruct",  # 使用7B版本而不是72B
            url='https://api-inference.modelscope.cn/v1/',
            api_key=api_key,
            timeout=60  # 减少超时时间
        )
        
        print(Fore.GREEN + "模型实例创建成功！")
        return model
        
    except Exception as e:
        print(Fore.RED + f"创建模型实例失败: {str(e)}")
        print(Fore.YELLOW + "尝试使用备用配置...")
        
        try:
            # 备用配置：使用不同的API端点
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                model_type="Qwen/Qwen2.5-7B-Instruct",
                url='https://api.modelscope.cn/v1/',  # 不同的API端点
                api_key=api_key,
                timeout=60
            )
            print(Fore.GREEN + "备用模型实例创建成功！")
            return model
        except Exception as e2:
            print(Fore.RED + f"备用配置也失败了: {str(e2)}")
            return None

def main(chat_turn_limit=5) -> None:  # 进一步减少对话轮数
    # 创建模型实例
    model = create_simple_model()
    if model is None:
        print(Fore.RED + "无法创建模型实例，程序退出")
        print(Fore.YELLOW + "建议检查：")
        print(Fore.YELLOW + "1. 网络连接是否正常")
        print(Fore.YELLOW + "2. API密钥是否正确")
        print(Fore.YELLOW + "3. 服务器是否可用")
        return
    
    task_prompt = "为股票市场开发一个简单的交易机器人"
    
    try:
        print(Fore.CYAN + "正在创建角色扮演会话...")
        role_play_session = RolePlaying(
            assistant_role_name="Python 程序员",
            assistant_agent_kwargs=dict(model=model),
            user_role_name="股票交易员",
            user_agent_kwargs=dict(model=model),
            task_prompt=task_prompt,
            with_task_specify=False,  # 关闭任务细化以减少复杂度
            output_language='中文'
        )
        
        print(Fore.GREEN + "角色扮演会话创建成功！")
        
        # 显示基本信息
        print(Fore.YELLOW + f"任务提示: {task_prompt}")
        print(Fore.CYAN + "开始对话...\n")
        
        # 开始对话循环
        n = 0
        input_msg = role_play_session.init_chat()
        
        while n < chat_turn_limit:
            n += 1
            print(Fore.MAGENTA + f"--- 第 {n} 轮对话 ---")
            
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
                print(Fore.BLUE + f"AI 用户: {user_response.msg.content}")
                print(Fore.GREEN + f"AI 助手: {assistant_response.msg.content}")
                print()
                
                # 检查是否完成任务
                if "CAMEL_TASK_DONE" in user_response.msg.content:
                    print(Fore.GREEN + "任务完成！")
                    break
                
                input_msg = assistant_response.msg
                
            except Exception as e:
                print(Fore.RED + f"对话过程中发生错误: {str(e)}")
                print(Fore.YELLOW + "跳过这一轮，继续下一轮...")
                continue
                
    except Exception as e:
        print(Fore.RED + f"创建角色扮演会话时发生错误: {str(e)}")
        return

if __name__ == "__main__":
    print(Fore.CYAN + "=== CAMEL AI 角色扮演系统 (简化版) ===")
    main() 